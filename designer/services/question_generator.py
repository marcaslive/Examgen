# designer/services/question_generator.py

import json
import logging
import os
import re
import time
import uuid
import requests
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Union

from django.conf import settings
from django.core.cache import cache

from ..models import Document
from .pdf_service import PDFService

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI_SDK = True
except ImportError:
    genai = None
    types = None
    HAS_GEMINI_SDK = False

# ============================================================
# CONFIG
# ============================================================
MIN_QUESTIONS = 10
MAX_QUESTIONS = 500
BATCH_LIMIT = 20

# Primary from settings, then known-good current IDs (May 2025+ style naming)
PRIMARY_MODEL = getattr(settings, "AI_MODEL", "gemini-2.0-flash")

# IMPORTANT: use IDs that exist on v1beta generateContent for your key.
# Avoid bare "gemini-1.5-pro" — often 404 now.
GEMINI_MODELS_CASCADE = [
    PRIMARY_MODEL,
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-8b",
    "gemini-pro-latest",
    # user said this is real for them — keep near end so it still tries
    "gemini-3.6-flash",
]

GEMINI_MAX_OUTPUT_TOKENS = 8192
GEMINI_INPUT_CHARS = 40000

HF_MAX_TOKENS = 1024
HF_INPUT_CHARS = 12000
HTTP_TIMEOUT = 35

MIN_REQUEST_INTERVAL = 1.5
MAX_FAIL_STREAK = 5
RATE_LIMIT_BACKOFF = 12

TASK_TIMEOUT = 7200
REVIEW_TIMEOUT = 7200

_default_hf = "Qwen/Qwen2.5-7B-Instruct,HuggingFaceH4/zephyr-7b-beta"
HF_CHAT_MODELS = [
    m.strip()
    for m in str(getattr(settings, "HF_CHAT_MODELS", _default_hf) or _default_hf).split(",")
    if m.strip()
]

# Cache key for discovered Gemini models (per process/day)
GEMINI_MODELS_CACHE_KEY = "qg:gemini:available_models"


def parse_keys(raw_val: str) -> List[str]:
    if not raw_val:
        return []
    return [k.strip() for k in str(raw_val).split(",") if k.strip()]


def _short_err(text: str, n: int = 180) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()[:n]


class GeminiRateLimiter:
    LAST_CALL_KEY = "qg:quota:last_call"
    DAY_KEY_PREFIX = "qg:quota:calls_day:"

    @classmethod
    def _day_key(cls):
        return cls.DAY_KEY_PREFIX + datetime.now(timezone.utc).strftime("%Y%m%d")

    @classmethod
    def get_status(cls) -> dict:
        now = time.time()
        last = float(cache.get(cls.LAST_CALL_KEY, 0) or 0)
        wait_sec = 0
        if last and (now - last) < MIN_REQUEST_INTERVAL:
            wait_sec = int(MIN_REQUEST_INTERVAL - (now - last)) + 1
        keys = parse_keys(getattr(settings, "GEMINI_API_KEY", "") or "")
        num_keys = max(1, len(keys))
        day_used = int(cache.get(cls._day_key(), 0) or 0)
        return {
            "rpd_used": day_used,
            "rpd_limit": 1500 * num_keys,
            "rpd_remaining": max(0, 1500 * num_keys - day_used),
            "rpm_used": 0,
            "rpm_limit": 60 * num_keys,
            "rpm_remaining": 60 * num_keys,
            "wait_seconds": wait_sec,
            "rpd_reset_seconds": 86400,
            "can_request": wait_sec == 0,
        }

    @classmethod
    def record_call(cls):
        cache.set(cls.LAST_CALL_KEY, time.time(), timeout=120)
        dk = cls._day_key()
        cache.set(dk, int(cache.get(dk, 0) or 0) + 1, timeout=172800)


def discover_gemini_models(api_key: str) -> List[str]:
    """Ask Google which models support generateContent for this key."""
    cached = cache.get(GEMINI_MODELS_CACHE_KEY)
    if isinstance(cached, list) and cached:
        return cached

    models = []
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            logger.warning(f"ListModels failed: {r.status_code} {_short_err(r.text)}")
            return []
        for m in r.json().get("models") or []:
            name = (m.get("name") or "").replace("models/", "")
            methods = m.get("supportedGenerationMethods") or []
            if name and "generateContent" in methods:
                models.append(name)
        # Prefer flash-like models first
        def rank(n: str):
            nlow = n.lower()
            score = 50
            if "flash" in nlow:
                score -= 20
            if "2.0" in nlow or "2.5" in nlow:
                score -= 10
            if "lite" in nlow:
                score -= 5
            if "pro" in nlow:
                score += 5
            if "embed" in nlow or "tts" in nlow or "image" in nlow:
                score += 100
            return score
        models = sorted(set(models), key=rank)
        if models:
            cache.set(GEMINI_MODELS_CACHE_KEY, models, timeout=3600)
            logger.info(f"Discovered Gemini models: {models[:12]}")
    except Exception as e:
        logger.warning(f"discover_gemini_models error: {e}")
    return models


class QuestionGenerator:
    def __init__(self):
        self.gemini_keys = parse_keys(getattr(settings, "GEMINI_API_KEY", "") or "")
        self.hf_token = (getattr(settings, "HUGGINGFACE_TOKEN", "") or "").strip()
        self.last_error = ""
        self.last_provider = ""
        self.gemini_error = ""

    @property
    def available(self) -> bool:
        return bool(self.gemini_keys) or bool(self.hf_token)

    def _text_to_str(self, text: Union[str, dict, list]) -> str:
        if isinstance(text, dict):
            return "\n\n".join(str(v) for v in text.values())
        if isinstance(text, list):
            return "\n\n".join(str(v) for v in text)
        return str(text or "")

    def _model_list_for_key(self, api_key: str) -> List[str]:
        """Cascade = primary + static list + discovered models (deduped)."""
        discovered = discover_gemini_models(api_key)
        ordered = []
        seen = set()
        for m in list(GEMINI_MODELS_CASCADE) + discovered:
            if not m or m in seen:
                continue
            # Skip obvious non-text models
            low = m.lower()
            if any(x in low for x in ("embed", "aqa", "tts", "image", "vision-preview-image")):
                continue
            seen.add(m)
            ordered.append(m)
        return ordered or list(GEMINI_MODELS_CASCADE)

    def generate(self, text, num_questions: int, batch_index: int = 0, total_batches: int = 1) -> List[dict]:
        if not self.available:
            self.last_error = "No GEMINI_API_KEY or HUGGINGFACE_TOKEN configured."
            return []

        full_text = self._text_to_str(text)
        prompt = self._build_prompt(
            full_text[:GEMINI_INPUT_CHARS], num_questions, batch_index, total_batches
        )

        safety_rest = [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        not_found_models = set()

        # -------- GEMINI PRIMARY --------
        for key in self.gemini_keys:
            models = self._model_list_for_key(key)
            for model in models:
                if model in not_found_models:
                    continue

                # A) SDK
                if HAS_GEMINI_SDK:
                    try:
                        logger.info(f"Gemini SDK → {model} batch {batch_index+1}/{total_batches}")
                        client = genai.Client(api_key=key)
                        config = types.GenerateContentConfig(
                            temperature=0.6,
                            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                            response_mime_type="application/json",
                        )
                        resp = client.models.generate_content(
                            model=model, contents=prompt, config=config
                        )
                        content = getattr(resp, "text", "") or ""
                        qs = self._parse_response(content)
                        if qs:
                            GeminiRateLimiter.record_call()
                            self.last_provider = f"gemini-sdk:{model}"
                            logger.info(f"✅ SDK {model}: {len(qs)} Qs")
                            return qs[:num_questions]
                    except Exception as e:
                        err = str(e)
                        self.gemini_error = f"SDK {model}: {_short_err(err)}"
                        self.last_error = self.gemini_error
                        logger.warning(self.last_error)
                        if "404" in err or "NOT_FOUND" in err or "is not found" in err:
                            not_found_models.add(model)
                            continue

                # B) REST
                try:
                    logger.info(f"Gemini REST → {model} batch {batch_index+1}/{total_batches}")
                    url = (
                        "https://generativelanguage.googleapis.com/v1beta/models/"
                        f"{model}:generateContent?key={key}"
                    )
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "safetySettings": safety_rest,
                        "generationConfig": {
                            "temperature": 0.6,
                            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
                            "responseMimeType": "application/json",
                        },
                    }
                    r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)

                    if r.status_code == 200:
                        data = r.json()
                        cands = data.get("candidates") or []
                        if not cands:
                            # often blocked / empty
                            fb = data.get("promptFeedback") or {}
                            self.gemini_error = f"REST {model}: no candidates { _short_err(str(fb)) }"
                            self.last_error = self.gemini_error
                            continue
                        parts = (cands[0].get("content") or {}).get("parts") or []
                        content = parts[0].get("text", "") if parts else ""
                        qs = self._parse_response(content)
                        if qs:
                            GeminiRateLimiter.record_call()
                            self.last_provider = f"gemini-rest:{model}"
                            logger.info(f"✅ REST {model}: {len(qs)} Qs")
                            return qs[:num_questions]
                        self.gemini_error = f"REST {model}: unparseable output"
                        self.last_error = self.gemini_error

                    elif r.status_code == 404:
                        # Model does not exist for this key/API — skip silently-fast
                        msg = _short_err(r.text)
                        not_found_models.add(model)
                        self.gemini_error = f"REST {model}: 404 not found"
                        self.last_error = self.gemini_error
                        logger.warning(f"Skipping missing model {model}: {msg}")
                        continue

                    elif r.status_code == 429:
                        self.gemini_error = f"REST {model}: 429 RATE LIMITED"
                        self.last_error = self.gemini_error
                        logger.warning(self.last_error)
                        # try next model/key
                        continue

                    else:
                        try:
                            msg = (r.json().get("error") or {}).get("message") or r.text
                        except Exception:
                            msg = r.text
                        self.gemini_error = f"REST {model} ({r.status_code}): {_short_err(msg)}"
                        self.last_error = self.gemini_error
                        logger.warning(self.last_error)
                        if "not found" in (msg or "").lower():
                            not_found_models.add(model)

                except requests.exceptions.RequestException as e:
                    self.gemini_error = f"REST {model}: connection ({e.__class__.__name__})"
                    self.last_error = self.gemini_error
                    logger.warning(self.last_error)
                except Exception as e:
                    self.gemini_error = f"REST {model}: {_short_err(str(e))}"
                    self.last_error = self.gemini_error
                    logger.warning(self.last_error)

        # -------- HF BACKUP --------
        if self.hf_token:
            qs = self._generate_hf(full_text, num_questions, batch_index, total_batches)
            if qs:
                return qs[:num_questions]

        if self.gemini_error:
            self.last_error = (
                f"Gemini failed ({self.gemini_error}) | "
                f"HF failed ({self.last_error if 'HF' in (self.last_error or '') else (self.last_error or 'n/a')})"
            )
        elif not self.last_error:
            self.last_error = "All AI providers failed."
        return []

    def _hf_msg(self, r: requests.Response) -> str:
        try:
            j = r.json()
            err = j.get("error", j)
            if isinstance(err, dict):
                return _short_err(str(err.get("message") or err))
            return _short_err(str(err))
        except Exception:
            return _short_err(r.text)

    def _generate_hf(self, full_text, num_questions, batch_index, total_batches) -> List[dict]:
        hf_n = min(num_questions, 8)
        prompt = self._build_prompt(
            full_text[:HF_INPUT_CHARS], hf_n, batch_index, total_batches
        )
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        url = "https://router.huggingface.co/v1/chat/completions"

        for model in HF_CHAT_MODELS:
            body = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert exam creator. Return valid JSON only. No markdown.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.6,
                "max_tokens": HF_MAX_TOKENS,
            }
            try:
                logger.info(f"HF chat → {model}")
                r = requests.post(url, headers=headers, json=body, timeout=HTTP_TIMEOUT)
                if r.status_code == 200:
                    content = (
                        r.json().get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    qs = self._parse_response(content)
                    if qs:
                        self.last_provider = f"hf:{model}"
                        logger.info(f"✅ HF {model}: {len(qs)} Qs")
                        return qs
                    self.last_error = f"HF {model.split('/')[-1]}: unparseable"
                elif r.status_code == 429:
                    self.last_error = f"HF {model.split('/')[-1]}: 429 rate limited"
                    return []
                else:
                    msg = self._hf_msg(r)
                    self.last_error = f"HF {model.split('/')[-1]} ({r.status_code}): {msg}"
                    logger.warning(self.last_error)
                    if any(s in msg.lower() for s in ("not supported", "not a chat", "does not exist")):
                        continue
            except requests.exceptions.RequestException:
                self.last_error = f"HF {model.split('/')[-1]}: timeout/connection"
            except Exception as e:
                self.last_error = f"HF {model.split('/')[-1]}: {e}"
        return []

    def _build_prompt(self, text, num_questions, batch_index, total_batches) -> str:
        extra = ""
        if total_batches > 1:
            extra = (
                f"This is batch {batch_index + 1} of {total_batches}. "
                f"Cover DIFFERENT facts than other batches.\n"
            )
        return f"""You are an expert exam creator. Generate EXACTLY {num_questions} multiple-choice questions from the study material.

{extra}RULES:
1. EXACTLY {num_questions} questions.
2. Explanations: one short sentence max.
3. correct_answer must be A, B, C, or D.
4. VALID JSON ONLY — no markdown.

JSON shape:
{{"questions":[{{"question_text":"...","option_a":"...","option_b":"...","option_c":"...","option_d":"...","correct_answer":"A","explanation":"...","source_page":1}}]}}

STUDY MATERIAL:
\"\"\"
{text}
\"\"\"
"""

    def _parse_response(self, content: str) -> List[dict]:
        content = (content or "").strip()
        if not content:
            return []
        m = re.search(r"```(?:json)?\s*(.*?)```", content, re.I | re.S)
        if m:
            content = m.group(1).strip()
        data = None
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            m2 = re.search(r"\{.*\}", content, re.S)
            if not m2:
                return self._parse_loose_objects(content)
            blob = m2.group(0)
            try:
                data = json.loads(blob)
            except json.JSONDecodeError:
                repaired = blob
                if repaired.count("[") > repaired.count("]"):
                    repaired += "]"
                if repaired.count("{") > repaired.count("}"):
                    repaired += "}"
                try:
                    data = json.loads(repaired)
                except json.JSONDecodeError:
                    return self._parse_loose_objects(content)
        raw = data.get("questions", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        return self._normalize_items(raw)

    def _parse_loose_objects(self, content: str) -> List[dict]:
        objs = re.findall(r"\{[^{}]*\"question_text\"[^{}]*\}", content, re.S)
        items = []
        for o in objs:
            try:
                items.append(json.loads(o))
            except json.JSONDecodeError:
                continue
        return self._normalize_items(items)

    def _normalize_items(self, raw) -> List[dict]:
        out = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            qt = str(item.get("question_text", "")).strip()
            oa = str(item.get("option_a", "")).strip()
            ob = str(item.get("option_b", "")).strip()
            oc = str(item.get("option_c", "")).strip()
            od = str(item.get("option_d", "")).strip()
            ans = str(item.get("correct_answer", "")).strip().upper()[:1]
            if not (qt and oa and ob and oc and od and ans in ("A", "B", "C", "D")):
                continue
            try:
                sp = int(item.get("source_page"))
            except (TypeError, ValueError):
                sp = None
            out.append({
                "question_text": qt,
                "option_a": oa,
                "option_b": ob,
                "option_c": oc,
                "option_d": od,
                "correct_answer": ans,
                "explanation": str(item.get("explanation", "")).strip(),
                "source_page": sp,
            })
        return out


class ExamGenerationManager:
    @staticmethod
    def get_quota_status() -> dict:
        return GeminiRateLimiter.get_status()

    @staticmethod
    def _task_key(task_id):
        return f"qg:task:{task_id}"

    @staticmethod
    def _review_key(review_id):
        return f"qg:review:{review_id}"

    @classmethod
    def _progress(cls, collected, target, batch_index, total_batches) -> int:
        if target <= 0:
            return 0
        by_count = (collected / target) * 100.0
        by_batch = (batch_index / max(total_batches, 1)) * 100.0
        p = max(by_count, by_batch * 0.9)
        return int(min(99, max(1, p))) if collected < target else 100

    @classmethod
    def start_generation(cls, request, payload: dict) -> Tuple[int, dict]:
        try:
            choice = str(payload.get("question_count_choice", "10")).strip().lower()
            raw_count = payload.get("custom_count") if choice == "custom" else choice
            try:
                count = max(MIN_QUESTIONS, min(MAX_QUESTIONS, int(raw_count)))
            except (TypeError, ValueError):
                return 400, {"success": False, "error": "Invalid question count."}

            doc_ids = payload.get("documents") or []
            if not doc_ids:
                return 400, {"success": False, "error": "Select at least one document."}

            batches, rem = [], count
            while rem > 0:
                b = min(BATCH_LIMIT, rem)
                batches.append(b)
                rem -= b

            try:
                text_by_page, source_map = cls._extract_text(request, payload)
            except Exception as e:
                return 400, {"success": False, "error": str(e)}

            if not text_by_page:
                return 400, {"success": False, "error": "No readable text found."}

            task_id = str(uuid.uuid4())
            state = {
                "task_id": task_id,
                "user_id": request.user.id,
                "target_count": count,
                "batch_sizes": batches,
                "total_batches": len(batches),
                "batch_index": 0,
                "fail_streak": 0,
                "collected": [],
                "text_by_page": text_by_page,
                "source_map": source_map,
                "document_ids": [str(d) for d in doc_ids],
            }
            cache.set(cls._task_key(task_id), state, timeout=TASK_TIMEOUT)
            request.session["gen_task_id"] = task_id
            request.session["generation_doc_ids"] = [str(d) for d in doc_ids]
            request.session.modified = True

            return 200, {
                "success": True,
                "plan": {
                    "total_questions": count,
                    "total_batches": len(batches),
                    "batch_size": BATCH_LIMIT,
                    "estimated_minutes": round(len(batches) * 0.35, 1),
                },
                "progress": 1,
                "message": f"Plan ready: {count} questions in {len(batches)} batches",
                "quota": cls.get_quota_status(),
            }
        except Exception as e:
            logger.exception("start_generation failed")
            return 500, {"success": False, "error": str(e)}

    @classmethod
    def process_next_batch(cls, request) -> Tuple[int, dict]:
        try:
            task_id = request.session.get("gen_task_id")
            if not task_id:
                return 400, {"success": False, "error": "No active session. Click Generate again."}

            state = cache.get(cls._task_key(task_id))
            if not state:
                return 410, {"success": False, "error": "Task expired. Start again."}

            target = state["target_count"]
            collected = state["collected"]
            total_batches = state["total_batches"]
            batch_index = state["batch_index"]

            if len(collected) >= target:
                return 200, cls._finish_generation(request, state)

            status = cls.get_quota_status()
            prog = cls._progress(len(collected), target, batch_index, total_batches)

            if status["wait_seconds"] > 0:
                return 429, {
                    "success": False,
                    "retryable": True,
                    "error": f"Pacing… {status['wait_seconds']}s",
                    "retry_after": status["wait_seconds"],
                    "progress": prog,
                    "total_so_far": len(collected),
                    "total_questions": target,
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "message": f"{len(collected)}/{target} — pause",
                    "quota": status,
                }

            batch_size = min(BATCH_LIMIT, target - len(collected))
            gen = QuestionGenerator()
            new_qs = gen.generate(
                state["text_by_page"],
                num_questions=batch_size,
                batch_index=batch_index,
                total_batches=total_batches,
            )

            if not new_qs:
                state["fail_streak"] = int(state.get("fail_streak", 0)) + 1
                cache.set(cls._task_key(task_id), state, timeout=TASK_TIMEOUT)
                err = gen.last_error or "AI returned no questions"
                is_rate = any(
                    x in err.lower()
                    for x in ("429", "rate limited", "resource_exhausted", "quota")
                )

                if state["fail_streak"] >= MAX_FAIL_STREAK and not is_rate:
                    if len(collected) >= MIN_QUESTIONS:
                        return 200, cls._finish_generation(request, state)
                    return 400, {
                        "success": False,
                        "retryable": False,
                        "error": f"Generation failed: {err}",
                        "progress": prog,
                        "total_so_far": len(collected),
                        "total_questions": target,
                        "quota": cls.get_quota_status(),
                    }

                return (429 if is_rate else 503), {
                    "success": False,
                    "retryable": True,
                    "retry_after": RATE_LIMIT_BACKOFF if is_rate else 3,
                    "error": f"{'Rate limited' if is_rate else 'Retrying'} ({state['fail_streak']}/{MAX_FAIL_STREAK}): {err[:240]}",
                    "progress": prog,
                    "total_so_far": len(collected),
                    "total_questions": target,
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "message": f"{len(collected)}/{target} — retrying",
                    "quota": cls.get_quota_status(),
                }

            collected.extend(new_qs)
            state["collected"] = collected
            state["batch_index"] = batch_index + 1
            state["fail_streak"] = 0
            cache.set(cls._task_key(task_id), state, timeout=TASK_TIMEOUT)

            prog = cls._progress(len(collected), target, state["batch_index"], total_batches)
            if len(collected) >= target:
                return 200, cls._finish_generation(request, state)

            return 200, {
                "success": True,
                "done": False,
                "batch_index": state["batch_index"],
                "total_batches": total_batches,
                "progress": prog,
                "total_so_far": len(collected),
                "total_questions": target,
                "message": f"Batch {batch_index+1}/{total_batches} — {len(collected)}/{target}",
                "provider": gen.last_provider,
                "quota": cls.get_quota_status(),
            }
        except Exception as exc:
            logger.exception("process_next_batch crashed")
            return 500, {"success": False, "error": str(exc)}

    @classmethod
    def _extract_text(cls, request, payload: dict) -> Tuple[Dict[int, str], Dict[str, dict]]:
        doc_ids = payload.get("documents") or []
        documents = list(
            Document.objects.filter(id__in=doc_ids, uploaded_by=request.user, status="ready")
        )
        if not documents:
            raise ValueError("No valid documents selected.")

        source_type = str(payload.get("source_type", "entire")).lower()
        text_by_page, source_map = {}, {}
        ref = 1
        for document in documents:
            if str(document.file_type).lower() != "pdf":
                continue
            try:
                path = document.file.path
            except Exception:
                raise ValueError(f"Document '{document.title}' path inaccessible.")
            if not os.path.exists(path):
                raise ValueError(f"File '{document.title}' missing. Re-upload it.")

            if source_type == "entire":
                pages = None
            elif source_type == "range":
                pages = PDFService.get_pages_for_range(
                    path, int(payload.get("page_from") or 1), int(payload.get("page_to") or 1)
                )
            elif source_type == "specific":
                pages = PDFService.parse_specific_pages(str(payload.get("specific_pages", "")))
            elif source_type == "random":
                pages = PDFService.get_random_pages(path, int(payload.get("random_page_count") or 5))
            else:
                pages = None

            extracted = PDFService.extract_text_from_pages(path, pages)
            for orig_page, page_text in extracted.items():
                if not str(page_text).strip():
                    continue
                text_by_page[ref] = f"[Document: {document.title}; page: {orig_page}]\n{page_text}"
                source_map[str(ref)] = {"document_id": str(document.id), "page": int(orig_page)}
                ref += 1
        return text_by_page, source_map

    @classmethod
    def _finish_generation(cls, request, state: dict) -> dict:
        target = state["target_count"]
        questions = state["collected"][:target]
        source_map = state.get("source_map", {})
        first = next(iter(source_map.values()), None)
        for q in questions:
            src = source_map.get(str(q.get("source_page", ""))) or first
            if src:
                q["source_document_id"] = src["document_id"]
                q["source_page"] = src["page"]

        review_id = str(uuid.uuid4())
        cache.set(
            cls._review_key(review_id),
            {"questions": questions, "document_ids": state.get("document_ids", [])},
            timeout=REVIEW_TIMEOUT,
        )
        request.session["review_task_id"] = review_id
        request.session["generated_questions"] = questions
        request.session["generation_doc_ids"] = state.get("document_ids", [])
        request.session.modified = True
        cache.delete(cls._task_key(state["task_id"]))
        request.session.pop("gen_task_id", None)
        return {
            "success": True,
            "done": True,
            "progress": 100,
            "count": len(questions),
            "total_so_far": len(questions),
            "total_questions": target,
            "message": f"Done — {len(questions)} questions ready",
            "quota": cls.get_quota_status(),
        }

    @classmethod
    def get_review_data(cls, request) -> dict:
        rid = request.session.get("review_task_id")
        if rid:
            data = cache.get(cls._review_key(rid))
            if data and data.get("questions"):
                return data
        return {
            "questions": request.session.get("generated_questions", []),
            "document_ids": request.session.get("generation_doc_ids", []),
        }

    @classmethod
    def clear_review_data(cls, request):
        rid = request.session.get("review_task_id")
        if rid:
            cache.delete(cls._review_key(rid))
        for k in ("review_task_id", "generated_questions", "generation_doc_ids", "gen_task_id"):
            request.session.pop(k, None)
        request.session.modified = True