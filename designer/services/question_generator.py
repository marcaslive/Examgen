# designer/services/question_generator.py

import json
import logging
import os
import re
import time
import uuid
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union

from django.conf import settings
from django.core.cache import cache

from ..models import Document
from .pdf_service import PDFService

logger = logging.getLogger(__name__)

# ============================================================
# SDK IMPORTS
# ============================================================
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    genai = None
    types = None
    HAS_GEMINI = False

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================
MIN_QUESTIONS = 10
MAX_QUESTIONS = 500
BATCH_LIMIT = 50

PRIMARY_MODEL = getattr(settings, "AI_MODEL", "gemini-3.6-flash")
GEMINI_MODELS_CASCADE = [
    PRIMARY_MODEL,
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

GEMINI_RPM_LIMIT = 5
GEMINI_RPD_LIMIT = 20
GEMINI_TPM_LIMIT = 250_000
MIN_REQUEST_INTERVAL = 12

TASK_TIMEOUT = 7200
REVIEW_TIMEOUT = 7200


def parse_keys(raw_val: str) -> List[str]:
    """Helper to cleanly parse comma-separated API keys."""
    if not raw_val:
        return []
    return [k.strip() for k in str(raw_val).split(",") if k.strip()]


# ============================================================
# REAL RATE LIMITER & QUOTA TRACKER
# ============================================================
class GeminiRateLimiter:
    PREFIX = "qg:quota:"

    @classmethod
    def _day_key(cls):
        now = datetime.now(timezone.utc)
        return f"{cls.PREFIX}day:{now.strftime('%Y%m%d')}"

    @classmethod
    def _calls_key(cls):
        return f"{cls.PREFIX}calls"

    @classmethod
    def _clean_calls(cls, values):
        now = time.time()
        out = []
        for v in values or []:
            try:
                v = float(v)
                if now - v < 60:
                    out.append(v)
            except (TypeError, ValueError):
                pass
        return out

    @classmethod
    def _seconds_until_midnight(cls):
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return max(1, int((tomorrow - now).total_seconds()))

    @classmethod
    def get_status(cls) -> dict:
        now = time.time()
        raw_keys = getattr(settings, "GEMINI_API_KEY", "") or ""
        keys = parse_keys(raw_keys)
        num_keys = max(1, len(keys))

        rpd_limit = GEMINI_RPD_LIMIT * num_keys
        rpm_limit = GEMINI_RPM_LIMIT * num_keys

        rpd_used = int(cache.get(cls._day_key(), 0) or 0)
        calls = cls._clean_calls(cache.get(cls._calls_key(), []))

        wait_sec = 0
        if calls:
            elapsed = now - max(calls)
            if elapsed < MIN_REQUEST_INTERVAL:
                wait_sec = max(wait_sec, int(MIN_REQUEST_INTERVAL - elapsed) + 1)
            if len(calls) >= rpm_limit:
                wait_sec = max(wait_sec, int(60 - (now - min(calls))) + 1)

        rpd_remaining = max(0, rpd_limit - rpd_used)
        rpm_remaining = max(0, rpm_limit - len(calls))

        return {
            "rpd_used": rpd_used,
            "rpd_limit": rpd_limit,
            "rpd_remaining": rpd_remaining,
            "rpm_used": len(calls),
            "rpm_limit": rpm_limit,
            "rpm_remaining": rpm_remaining,
            "wait_seconds": wait_sec,
            "rpd_reset_seconds": cls._seconds_until_midnight(),
            "can_request": (rpd_remaining > 0 and wait_sec == 0),
        }

    @classmethod
    def record_call(cls):
        now = time.time()
        day_key = cls._day_key()
        calls_key = cls._calls_key()

        calls = cls._clean_calls(cache.get(calls_key, []))
        calls.append(now)

        curr_rpd = int(cache.get(day_key, 0) or 0)
        cache.set(day_key, curr_rpd + 1, timeout=172800)
        cache.set(calls_key, calls, timeout=120)


# ============================================================
# MULTI-ENGINE AI GENERATOR
# ============================================================
class QuestionGenerator:
    """Primary: Gemini (3.6 -> 2.0 -> 1.5). Backup: Hugging Face (Qwen 2.5 72B)."""

    def __init__(self):
        raw_key = getattr(settings, "GEMINI_API_KEY", "") or ""
        self.gemini_keys = parse_keys(raw_key)
        self.hf_token = getattr(settings, "HUGGINGFACE_TOKEN", "") or ""

    @property
    def available(self) -> bool:
        return (HAS_GEMINI and len(self.gemini_keys) > 0 and bool(self.gemini_keys[0])) or bool(self.hf_token)

    def generate(self, text: Union[str, dict], num_questions: int, batch_index: int = 0, total_batches: int = 1) -> List[dict]:
        if not self.available:
            logger.error("No API keys found for Gemini or Hugging Face.")
            return []

        # Fix: Convert dict or list from text extractor into string before slicing
        if isinstance(text, dict):
            full_text = "\n\n".join(str(v) for v in text.values())
        elif isinstance(text, list):
            full_text = "\n\n".join(str(v) for v in text)
        else:
            full_text = str(text or "")

        prepared_text = full_text[:22000]
        prompt = self._build_prompt(prepared_text, num_questions, batch_index, total_batches)

        # 1. PRIMARY: GEMINI CASCADE
        if HAS_GEMINI and self.gemini_keys:
            for key in self.gemini_keys:
                try:
                    client = genai.Client(api_key=key)
                except Exception as e:
                    logger.warning(f"Failed to create Gemini client: {e}")
                    continue

                for model in GEMINI_MODELS_CASCADE:
                    try:
                        logger.info(f"Generating batch {batch_index+1} with {model}...")
                        config = types.GenerateContentConfig(
                            temperature=0.7,
                            max_output_tokens=8192,
                            response_mime_type="application/json",
                            safety_settings=[
                                types.SafetySetting(category="HATE_SPEECH", threshold="OFF"),
                                types.SafetySetting(category="HARASSMENT", threshold="OFF"),
                                types.SafetySetting(category="SEXUALLY_EXPLICIT", threshold="OFF"),
                                types.SafetySetting(category="DANGEROUS_CONTENT", threshold="OFF"),
                            ]
                        )
                        response = client.models.generate_content(model=model, contents=prompt, config=config)
                        content = getattr(response, "text", "") or ""
                        questions = self._parse_response(content)
                        if questions:
                            GeminiRateLimiter.record_call()
                            logger.info(f"✅ Success with {model}: Got {len(questions)} Qs")
                            return questions[:num_questions]
                    except Exception as exc:
                        logger.warning(f"Gemini {model} rate limited or failed: {exc}")
                        continue

        # 2. BACKUP: HUGGING FACE (Qwen 2.5 72B)
        if self.hf_token:
            try:
                logger.info("⚡ Gemini limit reached. Falling back to Hugging Face...")
                url = "https://router.huggingface.co/hf-inference/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.hf_token}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "Qwen/Qwen2.5-72B-Instruct",
                    "messages": [
                        {"role": "system", "content": "You are an expert exam question creator. Return valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4096,
                }
                res = requests.post(url, headers=headers, json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    content = data["choices"][0]["message"]["content"]
                    questions = self._parse_response(content)
                    if questions:
                        logger.info(f"✅ Hugging Face succeeded: Got {len(questions)} Qs")
                        return questions[:num_questions]
            except Exception as e:
                logger.error(f"Hugging Face backup error: {e}")

        return []

    def _build_prompt(self, text: str, num_questions: int, batch_index: int, total_batches: int) -> str:
        extra = f"This is batch {batch_index + 1} of {total_batches}. Do NOT repeat concepts from earlier batches.\n" if total_batches > 1 else ""
        return f"""You are an expert exam creator. Generate EXACTLY {num_questions} multiple-choice questions from the study material below.
{extra}
CRITICAL RULES:
1. Generate EXACTLY {num_questions} questions.
2. Keep explanations SHORT (1 concise sentence maximum) to ensure full output.
3. Return valid JSON ONLY. Do NOT use markdown code blocks or conversational text.

Required JSON Structure:
{{"questions": [{{"question_text": "...", "option_a": "...", "option_b": "...", "option_c": "...", "option_d": "...", "correct_answer": "A", "explanation": "...", "source_page": 1}}]}}

STUDY MATERIAL:
\"\"\"
{text}
\"\"\"
"""

    def _parse_response(self, content: str) -> List[dict]:
        content = (content or "").strip()
        m = re.search(r"```(?:json)?\s*(.*?)```", content, re.I | re.S)
        if m:
            content = m.group(1).strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            m2 = re.search(r"\{.*\}", content, re.S)
            if not m2:
                return []
            try:
                data = json.loads(m2.group(0))
            except json.JSONDecodeError:
                return []

        raw = data.get("questions", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        out = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            qt = str(item.get("question_text", "")).strip()
            oa = str(item.get("option_a", "")).strip()
            ob = str(item.get("option_b", "")).strip()
            oc = str(item.get("option_c", "")).strip()
            od = str(item.get("option_d", "")).strip()
            ans = str(item.get("correct_answer", "")).strip().upper()

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


# ============================================================
# EXAM GENERATION MANAGER (ACCUMULATOR ARCHITECTURE)
# ============================================================
class ExamGenerationManager:
    @staticmethod
    def get_quota_status() -> dict:
        return GeminiRateLimiter.get_status()

    @staticmethod
    def _task_key(task_id): return f"qg:task:{task_id}"
    @staticmethod
    def _review_key(review_id): return f"qg:review:{review_id}"

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

            status = cls.get_quota_status()
            batches_needed = (count // BATCH_LIMIT) + (1 if count % BATCH_LIMIT else 0)

            if status["rpd_remaining"] < batches_needed:
                return 429, {
                    "success": False,
                    "error": f"Daily limit exceeded. Need {batches_needed} requests, but only {status['rpd_remaining']} left today.",
                    "retry_after": status["rpd_reset_seconds"],
                    "quota": status,
                }

            if status["wait_seconds"] > 0:
                return 429, {
                    "success": False,
                    "error": f"Rate limit cooldown active. Please wait {status['wait_seconds']} seconds.",
                    "retry_after": status["wait_seconds"],
                    "quota": status,
                }

            batches = []
            rem = count
            while rem > 0:
                b = min(BATCH_LIMIT, rem)
                batches.append(b)
                rem -= b

            try:
                text_by_page, source_map = cls._extract_text(request, payload)
            except Exception as e:
                return 400, {"success": False, "error": str(e)}

            if not text_by_page:
                return 400, {"success": False, "error": "No readable text found in selected pages."}

            task_id = str(uuid.uuid4())
            state = {
                "task_id": task_id,
                "user_id": request.user.id,
                "target_count": count,
                "batch_sizes": batches,
                "total_batches": len(batches),
                "batch_index": 0,
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
                    "estimated_minutes": round(len(batches) * 0.3, 1),
                },
                "quota": status,
            }
        except Exception as e:
            logger.exception("start_generation crashed")
            return 500, {"success": False, "error": f"Initialization failed: {str(e)}"}

    @classmethod
    def process_next_batch(cls, request) -> Tuple[int, dict]:
        try:
            task_id = request.session.get("gen_task_id")
            if not task_id:
                return 400, {"success": False, "error": "No active session."}

            state = cache.get(cls._task_key(task_id))
            if not state:
                return 410, {"success": False, "error": "Task expired. Start again."}

            target = state["target_count"]
            collected = state["collected"]

            if len(collected) >= target:
                return 200, cls._finish_generation(request, state)

            status = cls.get_quota_status()
            if status["wait_seconds"] > 0:
                return 429, {
                    "success": False,
                    "retryable": True,
                    "error": f"Rate limit cooldown active. Resuming in {status['wait_seconds']}s...",
                    "retry_after": status["wait_seconds"],
                    "progress": int((len(collected) / target) * 100),
                    "total_so_far": len(collected),
                    "quota": status,
                }

            still_needed = target - len(collected)
            batch_size = min(BATCH_LIMIT, still_needed)

            gen = QuestionGenerator()
            new_questions = gen.generate(
                state["text_by_page"],
                num_questions=batch_size,
                batch_index=state["batch_index"],
                total_batches=state["total_batches"]
            )

            if not new_questions:
                return 503, {
                    "success": False,
                    "retryable": True,
                    "retry_after": 4,
                    "error": "AI engines busy. Retrying batch...",
                    "progress": int((len(collected) / target) * 100),
                    "total_so_far": len(collected),
                    "quota": cls.get_quota_status(),
                }

            collected.extend(new_questions)
            state["collected"] = collected
            state["batch_index"] += 1
            cache.set(cls._task_key(task_id), state, timeout=TASK_TIMEOUT)

            progress = int(min(100, (len(collected) / target) * 100))
            if len(collected) >= target:
                return 200, cls._finish_generation(request, state)

            return 200, {
                "success": True,
                "done": False,
                "batch_index": state["batch_index"],
                "total_batches": state["total_batches"],
                "progress": progress,
                "total_so_far": len(collected),
                "total_questions": target,
                "quota": cls.get_quota_status(),
            }
        except Exception as exc:
            logger.exception("process_next_batch crashed")
            return 500, {"success": False, "error": f"Generation error: {str(exc)}"}

    @classmethod
    def _extract_text(cls, request, payload: dict) -> Tuple[Dict[int, str], Dict[str, dict]]:
        doc_ids = payload.get("documents") or []
        documents = list(Document.objects.filter(id__in=doc_ids, uploaded_by=request.user, status="ready"))
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
                raise ValueError(f"Document '{document.title}' file path is inaccessible.")

            if not os.path.exists(path):
                raise ValueError(f"File '{document.title}' is missing from server storage. Please re-upload it.")

            if source_type == "entire":
                pages = None
            elif source_type == "range":
                pages = PDFService.get_pages_for_range(path, int(payload.get("page_from") or 1), int(payload.get("page_to") or 1))
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
            key = str(q.get("source_page", ""))
            src = source_map.get(key) or first
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
            "quota": cls.get_quota_status(),
        }

    @classmethod
    def get_review_data(cls, request) -> dict:
        review_id = request.session.get("review_task_id")
        if review_id:
            data = cache.get(cls._review_key(review_id))
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