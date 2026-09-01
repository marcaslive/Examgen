# designer/services/question_generator.py

import json
import logging
import os
import re
import time
import uuid
import hashlib
import requests
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Union, Set

from django.conf import settings
from django.core.cache import cache

from ..models import Document
from .pdf_service import PDFService

logger = logging.getLogger(__name__)

# ============================================================
# BULLETPROOF CONFIGURATION
# ============================================================
MIN_QUESTIONS = 10
MAX_QUESTIONS = 500
BATCH_LIMIT = 15

MIN_REQUEST_INTERVAL = 4.2
MAX_FAIL_STREAK = 8
RATE_LIMIT_BACKOFF = 10

GEMINI_MAX_OUTPUT_TOKENS = 8192
GEMINI_INPUT_CHARS = 35000

HF_MAX_TOKENS = 1024
HF_INPUT_CHARS = 12000

HTTP_TIMEOUT = 45
GENERATE_DEADLINE_SEC = 90

TASK_TIMEOUT = 7200
REVIEW_TIMEOUT = 7200

WORDS_PER_QUESTION = 40
MIN_WORDS_FOR_MIN_QUESTIONS = 80
CHARS_PER_WORD_EST = 5

_default_hf = "Qwen/Qwen2.5-7B-Instruct,HuggingFaceH4/zephyr-7b-beta"
HF_CHAT_MODELS = [
    m.strip()
    for m in str(getattr(settings, "HF_CHAT_MODELS", _default_hf) or _default_hf).split(",")
    if m.strip()
]

GEMINI_NOT_FOUND_CACHE_KEY = "qg:gemini:not_found_models"

# ==============================================================
# GEMINI MODEL PRIORITY:
# 1. Gemini 3.5 Flash Lite (Primary)
# 2. Gemini 3.1 Flash Lite (Backup)
# 3. Gemini 2.0 Flash / 1.5 Flash (Safety nets)
# ==============================================================
PRIMARY_MODEL = getattr(settings, "AI_MODEL", "gemini-3.5-flash-lite")

_KNOWN_GOOD = [
    "gemini-3.5-flash-lite",  # Primary
    "gemini-3.1-flash-lite",  # Backup
    "gemini-2.0-flash",       # Fallback 1
    "gemini-1.5-flash",       # Fallback 2
    "gemini-1.5-flash-8b",    # Fallback 3
]

GEMINI_MODELS_CASCADE = []
for m in [PRIMARY_MODEL] + _KNOWN_GOOD:
    if not m or m in GEMINI_MODELS_CASCADE:
        continue
    if m.lower() in ("gemini-3.6-flash",):
        continue
    GEMINI_MODELS_CASCADE.append(m)

if not GEMINI_MODELS_CASCADE:
    GEMINI_MODELS_CASCADE = list(_KNOWN_GOOD)

META_QUESTION_PATTERNS = [
    r"\baccording to the (text|passage|document|material|pdf|excerpt)\b",
    r"\bin the (text|passage|document|material|pdf|excerpt|section|chapter)\b",
    r"\bfrom the (text|passage|document|material|pdf|excerpt|section)\b",
    r"\bbased on the (text|passage|document|material|pdf|excerpt)\b",
    r"\bas (stated|mentioned|discussed|described|noted) in the (text|passage|document|section)\b",
    r"\bthe author (states|says|mentions|argues)\b",
    r"\bwhich (section|page|chapter|paragraph)\b",
    r"\bon page \d+\b",
    r"\bin section\b",
    r"\bwhat does the (text|passage|document) say\b",
    r"\baccording to the given\b",
    r"\bthe study material\b",
    r"\bthe provided (text|material|document)\b",
]

def parse_keys(raw_val: str) -> List[str]:
    if not raw_val:
        return []
    return [k.strip() for k in str(raw_val).split(",") if k.strip()]

def _short_err(text: str, n: int = 180) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()[:n]

def estimate_max_questions_from_text(text: str) -> dict:
    raw = (text or "").strip()
    cleaned = re.sub(r"\s+", " ", raw)
    chars = len(cleaned)
    words = len([w for w in re.findall(r"[A-Za-z0-9]{2,}", cleaned)])
    if words < 5 and chars > 0:
        words = max(1, chars // CHARS_PER_WORD_EST)

    if words < 30:
        max_q = 5
    elif words < MIN_WORDS_FOR_MIN_QUESTIONS:
        max_q = max(5, words // 12)
    else:
        by_words = words // WORDS_PER_QUESTION
        by_chars = chars // 200
        max_q = max(by_words, min(by_chars, by_words + 15))

    max_q = int(max(5, min(MAX_QUESTIONS, max_q)))
    density_cap = max(5, min(MAX_QUESTIONS, words // 20))
    max_q = min(max_q, density_cap) if words >= 30 else max_q

    return {
        "word_count": words,
        "char_count": chars,
        "max_questions": max_q,
        "min_questions": MIN_QUESTIONS if max_q >= MIN_QUESTIONS else max(1, max_q),
    }

def normalize_question_key(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    stop = {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
        "what", "which", "who", "when", "where", "how", "does", "do", "did",
        "following", "below", "above", "most", "best", "correct",
    }
    tokens = [w for w in t.split() if w not in stop and len(w) > 1]
    return " ".join(tokens[:28])

def question_fingerprint(text: str) -> str:
    return hashlib.sha1(normalize_question_key(text).encode("utf-8")).hexdigest()[:16]

def is_meta_question(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t, re.I) for p in META_QUESTION_PATTERNS)

def is_near_duplicate(a: str, b: str, threshold: float = 0.85) -> bool:
    ta = set(normalize_question_key(a).split())
    tb = set(normalize_question_key(b).split())
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    union = len(ta | tb)
    return union > 0 and (inter / union) >= threshold

def _get_not_found_models() -> Set[str]:
    cached = cache.get(GEMINI_NOT_FOUND_CACHE_KEY) or []
    return set(cached) if isinstance(cached, list) else set()

def _remember_not_found_model(model: str) -> None:
    if not model:
        return
    found = _get_not_found_models()
    if model in found:
        return
    found.add(model)
    cache.set(GEMINI_NOT_FOUND_CACHE_KEY, list(found), timeout=3600)

def _is_model_busy(model: str) -> bool:
    return bool(cache.get(f"qg:busy:model:{model}"))

def _mark_model_busy(model: str, seconds: int = 15) -> None:
    cache.set(f"qg:busy:model:{model}", True, timeout=seconds)

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
            wait_sec = max(1, int(MIN_REQUEST_INTERVAL - (now - last)) + 1)
        keys = parse_keys(getattr(settings, "GEMINI_API_KEY", "") or "")
        num_keys = max(1, len(keys))
        day_used = int(cache.get(cls._day_key(), 0) or 0)
        return {
            "rpd_used": day_used,
            "rpd_limit": 1500 * num_keys,
            "rpd_remaining": max(0, 1500 * num_keys - day_used),
            "rpm_used": 0,
            "rpm_limit": 15 * num_keys,
            "rpm_remaining": 15 * num_keys,
            "wait_seconds": wait_sec,
            "rpd_reset_seconds": 86400,
            "can_request": wait_sec == 0,
        }

    @classmethod
    def record_call(cls):
        cache.set(cls.LAST_CALL_KEY, time.time(), timeout=60)
        dk = cls._day_key()
        cache.set(dk, int(cache.get(dk, 0) or 0) + 1, timeout=172800)


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

    def _models(self) -> List[str]:
        not_found = _get_not_found_models()
        out = [m for m in GEMINI_MODELS_CASCADE if m not in not_found and not _is_model_busy(m)]
        return out if out else [m for m in GEMINI_MODELS_CASCADE if m not in not_found]

    def generate(
        self,
        text,
        num_questions: int,
        batch_index: int = 0,
        total_batches: int = 1,
        avoid_questions: List[str] = None,
    ) -> List[dict]:
        if not self.available:
            self.last_error = "No API keys configured."
            return []

        full_text = self._text_to_str(text)
        avoid_questions = avoid_questions or []
        prompt = self._build_prompt(
            full_text[:GEMINI_INPUT_CHARS],
            num_questions,
            batch_index,
            total_batches,
            avoid_questions,
        )

        safety_rest = [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        deadline = time.time() + GENERATE_DEADLINE_SEC
        attempts = 0

        for key in self.gemini_keys:
            if not key:
                continue

            for model in self._models():
                if time.time() >= deadline or attempts >= 5:
                    break

                left = deadline - time.time()
                if left < 5:
                    break

                call_timeout = max(10, min(HTTP_TIMEOUT, int(left - 2)))
                attempts += 1

                try:
                    logger.info("Gemini REST → %s batch %s/%s", model, batch_index + 1, total_batches)
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                    
                    r = requests.post(
                        url,
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "safetySettings": safety_rest,
                            "generationConfig": {
                                "temperature": 0.7,
                                "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
                                "responseMimeType": "application/json",
                            },
                        },
                        timeout=call_timeout,
                    )

                    if r.status_code == 200:
                        data = r.json()
                        cands = data.get("candidates") or []
                        if not cands:
                            self.gemini_error = f"{model}: Empty response from Google"
                            continue
                        
                        parts = (cands[0].get("content") or {}).get("parts") or []
                        content = parts[0].get("text", "") if parts else ""
                        qs = self._filter_quality(self._parse_response(content), avoid_questions)
                        
                        if qs:
                            GeminiRateLimiter.record_call()
                            self.last_provider = f"gemini-rest:{model}"
                            return qs[:num_questions]
                            
                        self.gemini_error = f"{model}: Returned invalid JSON format"
                        self.last_error = self.gemini_error

                    elif r.status_code == 404:
                        _remember_not_found_model(model)
                        self.gemini_error = f"{model} does not exist"
                        logger.warning(f"Model {model} returned 404, switching to next")
                        continue

                    elif r.status_code == 429:
                        _mark_model_busy(model, 15)
                        self.gemini_error = f"{model}: Rate limit hit"
                        logger.warning(f"Model {model} rate limited, switching to next")
                        break  # Try next key or backup model

                    else:
                        try:
                            msg = (r.json().get("error") or {}).get("message") or r.text
                        except Exception:
                            msg = r.text
                        self.gemini_error = f"HTTP {r.status_code}: {_short_err(msg)}"

                except requests.exceptions.Timeout:
                    self.gemini_error = f"{model}: Connection timed out"
                except Exception as e:
                    self.gemini_error = f"{model}: {_short_err(str(e))}"

        # ---------- HuggingFace Fallback ----------
        if self.hf_token and time.time() < deadline:
            qs = self._generate_hf(
                full_text, num_questions, batch_index, total_batches, avoid_questions, deadline
            )
            if qs:
                return qs[:num_questions]

        if not self.last_error:
            self.last_error = self.gemini_error or "All AI models failed to generate valid questions."
        return []

    def _filter_quality(self, qs: List[dict], avoid_questions: List[str]) -> List[dict]:
        out = []
        batch_fps: Set[str] = set()
        avoid_fps = {question_fingerprint(q) for q in (avoid_questions or []) if q}
        avoid_texts = list(avoid_questions or [])
        for q in qs or []:
            qt = (q.get("question_text") or "").strip()
            if not qt or len(qt) < 12 or is_meta_question(qt):
                continue
            fp = question_fingerprint(qt)
            if fp in batch_fps or fp in avoid_fps:
                continue
            if any(is_near_duplicate(qt, p) for p in avoid_texts):
                continue
            if any(is_near_duplicate(qt, x.get("question_text", "")) for x in out):
                continue
            batch_fps.add(fp)
            out.append(q)
        return out

    def _generate_hf(self, full_text, num_questions, batch_index, total_batches, avoid_questions, deadline=None):
        hf_n = min(num_questions, 8)
        prompt = self._build_prompt(full_text[:HF_INPUT_CHARS], hf_n, batch_index, total_batches, avoid_questions)
        headers = {"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"}
        for model in HF_CHAT_MODELS:
            if deadline and time.time() >= deadline:
                break
            try:
                r = requests.post(
                    "https://router.huggingface.co/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "Exam writer. Valid JSON only."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.7,
                        "max_tokens": HF_MAX_TOKENS,
                    },
                    timeout=20,
                )
                if r.status_code == 200:
                    content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    qs = self._filter_quality(self._parse_response(content), avoid_questions)
                    if qs:
                        self.last_provider = f"hf:{model}"
                        return qs
            except Exception:
                continue
        return []

    def _build_prompt(self, text, num_questions, batch_index, total_batches, avoid_questions=None):
        extra = ""
        if total_batches > 1:
            extra = f"Batch {batch_index + 1}/{total_batches}. Different facts than other batches.\n"
        avoid_block = ""
        if avoid_questions:
            lines = "\n".join(f"- {q[:100]}" for q in avoid_questions[-20:] if q)
            avoid_block = f"\nDo NOT repeat:\n{lines}\n"
        return f"""Create EXACTLY {num_questions} MCQs from the material.
{extra}{avoid_block}
Rules: standalone questions; no "according to the text"; options A-D; JSON only.
{{"questions":[{{"question_text":"...","option_a":"...","option_b":"...","option_c":"...","option_d":"...","correct_answer":"A","explanation":"...","source_page":1}}]}}
MATERIAL:
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
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            m2 = re.search(r"\{.*\}", content, re.S)
            if not m2:
                return self._parse_loose(content)
            try:
                data = json.loads(m2.group(0))
            except json.JSONDecodeError:
                return self._parse_loose(content)
        raw = data.get("questions", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        return self._normalize(raw)

    def _parse_loose(self, content: str) -> List[dict]:
        items = []
        for o in re.findall(r"\{[^{}]*\"question_text\"[^{}]*\}", content, re.S):
            try:
                items.append(json.loads(o))
            except json.JSONDecodeError:
                pass
        return self._normalize(items)

    def _normalize(self, raw) -> List[dict]:
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
            if not (qt and oa and ob and oc and od and ans in "ABCD"):
                continue
            if is_meta_question(qt):
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
    def _merge_unique(cls, existing: List[dict], new_qs: List[dict]) -> List[dict]:
        merged = list(existing)
        texts = [q.get("question_text", "") for q in merged]
        fps = {question_fingerprint(t) for t in texts if t}
        for q in new_qs or []:
            qt = (q.get("question_text") or "").strip()
            if not qt or is_meta_question(qt):
                continue
            fp = question_fingerprint(qt)
            if fp in fps:
                continue
            if any(is_near_duplicate(qt, p) for p in texts):
                continue
            fps.add(fp)
            texts.append(qt)
            merged.append(q)
        return merged

    @classmethod
    def start_generation(cls, request, payload: dict) -> Tuple[int, dict]:
        try:
            choice = str(payload.get("question_count_choice", "10")).strip().lower()
            raw = payload.get("custom_count") if choice == "custom" else choice
            try:
                requested = int(raw)
            except (TypeError, ValueError):
                return 400, {"success": False, "error": "Invalid question count."}

            doc_ids = payload.get("documents") or []
            if not doc_ids:
                return 400, {"success": False, "error": "Select at least one document."}

            try:
                text_by_page, source_map = cls._extract_text(request, payload)
            except Exception as e:
                return 400, {"success": False, "error": str(e)}

            if not text_by_page:
                return 400, {"success": False, "error": "No readable text found in document."}

            full_text = "\n\n".join(str(v) for v in text_by_page.values())
            capacity = estimate_max_questions_from_text(full_text)
            max_possible = capacity["max_questions"]

            if max_possible < 1:
                return 400, {"success": False, "error": "Material too short to generate questions.", "material": capacity}

            if requested > max_possible:
                requested = max_possible

            count = max(1, min(MAX_QUESTIONS, requested))

            batches, rem = [], count
            while rem > 0:
                b = min(BATCH_LIMIT, rem)
                batches.append(b)
                rem -= b

            task_id = str(uuid.uuid4())
            state = {
                "task_id": task_id,
                "user_id": getattr(request.user, "id", None),
                "target_count": count,
                "batch_sizes": batches,
                "total_batches": len(batches),
                "batch_index": 0,
                "fail_streak": 0,
                "collected": [],
                "text_by_page": text_by_page,
                "source_map": source_map,
                "document_ids": [str(d) for d in doc_ids],
                "max_questions_for_material": max_possible,
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
                },
                "max_questions_for_material": max_possible,
                "material": capacity,
                "progress": 0,
                "message": f"Plan ready: {count} questions",
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
                return 400, {"success": False, "error": "Session expired. Click Generate again."}

            state = cache.get(cls._task_key(task_id))
            if not state:
                return 410, {"success": False, "error": "Task expired. Start again."}

            target = state["target_count"]
            collected = state["collected"]
            total_batches = state["total_batches"]
            batch_index = state["batch_index"]
            max_mat = state.get("max_questions_for_material", target)

            if len(collected) >= target:
                return 200, cls._finish_generation(request, state)

            status = cls.get_quota_status()
            if status["wait_seconds"] > 0:
                pct = int(min(99, max(1, (len(collected) / max(target, 1)) * 100)))
                return 429, {
                    "success": False,
                    "retryable": True,
                    "retry_after": status["wait_seconds"],
                    "error": f"Pacing {status['wait_seconds']}s",
                    "progress": pct,
                    "total_so_far": len(collected),
                    "total_questions": target,
                    "message": f"{len(collected)}/{target}",
                    "quota": status,
                }

            still = target - len(collected)
            ask_n = min(BATCH_LIMIT, still + 3)
            avoid = [q.get("question_text", "") for q in collected]

            gen = QuestionGenerator()
            try:
                new_qs = gen.generate(
                    state["text_by_page"],
                    num_questions=ask_n,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    avoid_questions=avoid,
                )
            except Exception as e:
                logger.exception("generate raised")
                new_qs = []
                gen.last_error = _short_err(str(e))

            if new_qs:
                before = len(collected)
                collected = cls._merge_unique(collected, new_qs)
                gained = len(collected) - before
            else:
                gained = 0

            if gained <= 0:
                state["fail_streak"] = int(state.get("fail_streak", 0)) + 1
                cache.set(cls._task_key(task_id), state, timeout=TASK_TIMEOUT)
                
                err = gen.last_error or "AI returned duplicate or invalid questions."
                is_rate = any(x in err.lower() for x in ("429", "rate", "quota", "busy"))

                if len(collected) >= target:
                    state["collected"] = collected
                    return 200, cls._finish_generation(request, state)

                if state["fail_streak"] >= 3 and len(collected) >= max(5, int(target * 0.6)):
                    state["collected"] = collected
                    state["target_count"] = len(collected)
                    cache.set(cls._task_key(task_id), state, timeout=TASK_TIMEOUT)
                    return 200, cls._finish_generation(request, state)

                if state["fail_streak"] >= MAX_FAIL_STREAK:
                    if len(collected) >= 5:
                        state["collected"] = collected
                        state["target_count"] = len(collected)
                        return 200, cls._finish_generation(request, state)
                    return 400, {
                        "success": False,
                        "retryable": False,
                        "error": f"Generation failed: {err}",
                        "total_so_far": len(collected),
                        "total_questions": target,
                        "progress": int((len(collected) / max(target, 1)) * 100),
                    }

                pct = int(min(99, max(1, (len(collected) / max(target, 1)) * 100)))
                return 429, {
                    "success": False,
                    "retryable": True,
                    "retry_after": RATE_LIMIT_BACKOFF if is_rate else 2,
                    "error": f"Retrying... ({err})",
                    "progress": pct,
                    "total_so_far": len(collected),
                    "total_questions": target,
                    "message": f"{len(collected)}/{target}",
                    "quota": cls.get_quota_status(),
                }

            state["collected"] = collected
            state["batch_index"] = batch_index + 1
            state["fail_streak"] = 0
            cache.set(cls._task_key(task_id), state, timeout=TASK_TIMEOUT)

            if len(collected) >= target:
                return 200, cls._finish_generation(request, state)

            pct = int(min(99, max(1, (len(collected) / max(target, 1)) * 100)))
            return 200, {
                "success": True,
                "done": False,
                "progress": pct,
                "total_so_far": len(collected),
                "total_questions": target,
                "batch_index": state["batch_index"],
                "total_batches": total_batches,
                "max_questions_for_material": max_mat,
                "message": f"{len(collected)}/{target} unique",
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
            try:
                path = document.file.path
            except Exception:
                continue
            if not path or not os.path.exists(path):
                continue

            ext = str(document.file_type or "").lower().strip().lstrip(".")
            if not ext and getattr(document, "original_filename", None):
                ext = document.original_filename.split(".")[-1].lower()

            if ext in ("pdf", "application/pdf") or not ext:
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
                    title = getattr(document, "title", None) or getattr(document, "original_filename", "doc")
                    text_by_page[ref] = f"[Document: {title}; page: {orig_page}]\n{page_text}"
                    source_map[str(ref)] = {"document_id": str(document.id), "page": int(orig_page)}
                    ref += 1

            elif ext in ("txt", "text"):
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if content.strip():
                        title = getattr(document, "title", None) or "doc"
                        text_by_page[ref] = f"[Document: {title}]\n{content}"
                        source_map[str(ref)] = {"document_id": str(document.id), "page": 1}
                        ref += 1
                except Exception:
                    pass

        return text_by_page, source_map

    @classmethod
    def _finish_generation(cls, request, state: dict) -> dict:
        target = state["target_count"]
        unique = cls._merge_unique([], state["collected"])
        questions = unique[:target]
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
            "total_questions": len(questions),
            "max_questions_for_material": state.get("max_questions_for_material"),
            "message": f"Done — {len(questions)} unique questions ready",
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