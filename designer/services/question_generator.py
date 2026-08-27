# designer/services/question_generator.py

import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from django.conf import settings
from django.core.cache import cache

from ..models import Document
from .pdf_service import PDFService

logger = logging.getLogger(__name__)

# ============================================================
# GEMINI SDK
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
# CONSTANTS
# ============================================================

MIN_QUESTIONS = 10
MAX_QUESTIONS = 500
BATCH_LIMIT = 50

GEMINI_RPM_LIMIT = 5
GEMINI_RPD_LIMIT = 20
GEMINI_TPM_LIMIT = 250_000

MIN_REQUEST_INTERVAL = 12
TASK_TIMEOUT = 7200
REVIEW_TIMEOUT = 7200

# ============================================================
# EXCEPTIONS
# ============================================================

class IncompleteBatchError(Exception):
    """Gemini did not return exactly the requested number of questions."""


class ProviderRateLimitError(Exception):
    """Gemini returned a rate-limit response."""

    def __init__(self, message, retry_after=60):
        super().__init__(message)
        self.retry_after = retry_after

# ============================================================
# BATCH PLAN
# ============================================================

def compute_batch_plan(num_questions: int) -> dict:
    num_questions = int(num_questions)

    if num_questions < MIN_QUESTIONS or num_questions > MAX_QUESTIONS:
        raise ValueError(
            f"Question count must be between {MIN_QUESTIONS} and {MAX_QUESTIONS}."
        )

    if num_questions <= BATCH_LIMIT:
        sizes = [num_questions]
    else:
        sizes = []
        remaining = num_questions
        while remaining > 0:
            current_size = min(BATCH_LIMIT, remaining)
            sizes.append(current_size)
            remaining -= current_size

    total_batches = len(sizes)
    estimated_seconds = (
        total_batches * 30
        + max(0, total_batches - 1) * MIN_REQUEST_INTERVAL
    )

    return {
        "total_questions": num_questions,
        "total_batches": total_batches,
        "batch_sizes": sizes,
        "estimated_minutes": round(estimated_seconds / 60, 1),
    }

# ============================================================
# RATE LIMITER
# ============================================================

class GeminiRateLimiter:
    PREFIX = "question_generation:quota:"
    LOCK_KEY = f"{PREFIX}lock"

    def _day_key(self):
        now = datetime.now(timezone.utc)
        return f"{self.PREFIX}day:{now.strftime('%Y%m%d')}"

    def _calls_key(self):
        return f"{self.PREFIX}calls"

    def _tokens_key(self):
        return f"{self.PREFIX}tokens"

    @staticmethod
    def _clean_calls(values):
        now = time.time()
        cleaned = []
        for value in values or []:
            try:
                value = float(value)
                if now - value < 60:
                    cleaned.append(value)
            except (TypeError, ValueError):
                continue
        return cleaned

    @staticmethod
    def _clean_tokens(values):
        now = time.time()
        cleaned = []
        for item in values or []:
            try:
                timestamp, token_count = item
                timestamp = float(timestamp)
                token_count = int(token_count)
                if now - timestamp < 60:
                    cleaned.append((timestamp, token_count))
            except (TypeError, ValueError):
                continue
        return cleaned

    @staticmethod
    def _seconds_until_midnight():
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return max(1, int((tomorrow - now).total_seconds()))

    def get_status(self) -> dict:
        now = time.time()
        calls = self._clean_calls(cache.get(self._calls_key(), []))
        tokens = self._clean_tokens(cache.get(self._tokens_key(), []))

        rpd_used = int(cache.get(self._day_key(), 0) or 0)
        tpm_used = sum(item[1] for item in tokens)

        wait_for_interval = 0
        if calls:
            elapsed = now - max(calls)
            if elapsed < MIN_REQUEST_INTERVAL:
                wait_for_interval = int(MIN_REQUEST_INTERVAL - elapsed) + 1

        wait_for_rpm = 0
        if len(calls) >= GEMINI_RPM_LIMIT:
            oldest_call = min(calls)
            wait_for_rpm = int(60 - (now - oldest_call)) + 1

        wait_seconds = max(wait_for_interval, wait_for_rpm)

        return {
            "rpd_used": rpd_used,
            "rpd_limit": GEMINI_RPD_LIMIT,
            "rpd_remaining": max(0, GEMINI_RPD_LIMIT - rpd_used),
            "rpm_used": len(calls),
            "rpm_limit": GEMINI_RPM_LIMIT,
            "rpm_remaining": max(0, GEMINI_RPM_LIMIT - len(calls)),
            "tpm_used": tpm_used,
            "tpm_limit": GEMINI_TPM_LIMIT,
            "wait_seconds": wait_seconds,
            "rpd_reset_seconds": self._seconds_until_midnight(),
            "can_request": (
                rpd_used < GEMINI_RPD_LIMIT
                and len(calls) < GEMINI_RPM_LIMIT
                and wait_seconds == 0
                and tpm_used < GEMINI_TPM_LIMIT
            ),
        }

    def can_start(self, batches_needed: int) -> Tuple[bool, str, int]:
        status = self.get_status()

        if status["rpd_remaining"] < batches_needed:
            reset_seconds = status["rpd_reset_seconds"]
            return (
                False,
                (
                    f"This generation needs {batches_needed} API requests, "
                    f"but only {status['rpd_remaining']} daily requests "
                    f"remain. Try again after the daily reset."
                ),
                reset_seconds,
            )

        return True, "", 0

    def acquire(self, estimated_tokens: int) -> Tuple[bool, str, int]:
        lock_id = str(uuid.uuid4())

        if not cache.add(self.LOCK_KEY, lock_id, timeout=10):
            return (
                False,
                "Another generation request is currently being processed.",
                3,
            )

        try:
            now = time.time()
            calls = self._clean_calls(cache.get(self._calls_key(), []))
            tokens = self._clean_tokens(cache.get(self._tokens_key(), []))
            rpd_used = int(cache.get(self._day_key(), 0) or 0)
            tpm_used = sum(item[1] for item in tokens)

            if rpd_used >= GEMINI_RPD_LIMIT:
                return (
                    False,
                    "The daily Gemini request limit has been reached.",
                    self._seconds_until_midnight(),
                )

            if calls:
                elapsed = now - max(calls)
                if elapsed < MIN_REQUEST_INTERVAL:
                    wait = int(MIN_REQUEST_INTERVAL - elapsed) + 1
                    return (
                        False,
                        "Please wait before sending the next request.",
                        wait,
                    )

            if len(calls) >= GEMINI_RPM_LIMIT:
                oldest_call = min(calls)
                wait = int(60 - (now - oldest_call)) + 1
                return (
                    False,
                    "The per-minute Gemini request limit has been reached.",
                    wait,
                )

            if tpm_used + estimated_tokens > GEMINI_TPM_LIMIT:
                wait = 61
                if tokens:
                    oldest_token_time = min(item[0] for item in tokens)
                    wait = max(1, int(60 - (now - oldest_token_time)) + 1)
                return (
                    False,
                    "The per-minute token limit has been reached.",
                    wait,
                )

            calls.append(now)
            tokens.append((now, int(estimated_tokens)))

            cache.set(self._calls_key(), calls, timeout=120)
            cache.set(self._tokens_key(), tokens, timeout=120)
            cache.set(self._day_key(), rpd_used + 1, timeout=172800)

            return True, "", 0

        finally:
            cache.delete(self.LOCK_KEY)

# ============================================================
# AI QUESTION GENERATOR
# ============================================================

class QuestionGenerator:
    def __init__(self):
        self.gemini_key = getattr(settings, "GEMINI_API_KEY", "")
        self.model_name = getattr(settings, "AI_MODEL", "gemini-3.6-flash")
        self.max_output_tokens = int(getattr(settings, "AI_MAX_OUTPUT_TOKENS", 8192))
        self.gemini_client = None

        if HAS_GEMINI and self.gemini_key:
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_key)
                logger.info("Gemini client ready: %s", self.model_name)
            except Exception:
                logger.exception("Gemini client initialization failed.")

    @property
    def available(self):
        return self.gemini_client is not None

    def generate(
        self,
        text_by_page,
        num_questions: int,
        batch_index: int = 0,
        total_batches: int = 1,
    ) -> List[dict]:
        num_questions = int(num_questions)

        if num_questions < 1 or num_questions > BATCH_LIMIT:
            raise ValueError(f"Each API batch must contain 1–{BATCH_LIMIT} questions.")

        if not self.available:
            raise RuntimeError("Gemini is not configured. Check GEMINI_API_KEY.")

        text = self._prepare_text(text_by_page)

        if not text.strip():
            raise ValueError("The selected source contains no readable text.")

        prompt = self._build_prompt(
            text=text,
            num_questions=num_questions,
            batch_index=batch_index,
            total_batches=total_batches,
        )

        try:
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=self.max_output_tokens,
                    response_mime_type="application/json",
                ),
            )
        except Exception as exc:
            message = str(exc).lower()
            if any(
                phrase in message
                for phrase in (
                    "429",
                    "resource_exhausted",
                    "rate limit",
                    "quota",
                    "too many requests",
                )
            ):
                raise ProviderRateLimitError(
                    "Gemini rate limit reached.",
                    retry_after=self._extract_retry_seconds(str(exc)),
                ) from exc
            raise

        content = getattr(response, "text", None)

        if not content:
            raise IncompleteBatchError("Gemini returned an empty response.")

        questions = self._parse_response(content)
        questions = questions[:num_questions]

        if len(questions) != num_questions:
            raise IncompleteBatchError(
                f"Gemini returned {len(questions)} valid questions, but {num_questions} were requested."
            )

        return questions

    def generate_questions(
        self,
        text_by_page,
        num_questions: int,
        source_document_id: Optional[str] = None,
    ) -> List[dict]:
        plan = compute_batch_plan(int(num_questions))
        all_questions = []

        for index, batch_size in enumerate(plan["batch_sizes"]):
            questions = self.generate(
                text_by_page=text_by_page,
                num_questions=batch_size,
                batch_index=index,
                total_batches=plan["total_batches"],
            )
            all_questions.extend(questions)

        return all_questions[:plan["total_questions"]]

    def _prepare_text(self, text_by_page) -> str:
        if isinstance(text_by_page, str):
            text = text_by_page
            if len(text) > 20_000:
                text = text[:20_000]
            return text

        parts = []
        try:
            items = sorted(text_by_page.items(), key=lambda item: int(item[0]))
        except (AttributeError, TypeError, ValueError):
            items = []

        for page_key, page_text in items:
            page_text = str(page_text).strip()
            if not page_text:
                continue
            if len(page_text) > 3_000:
                page_text = page_text[:3_000]
            parts.append(f"[Source reference {page_key}]\n{page_text}")

        combined = "\n\n".join(parts)
        if len(combined) > 20_000:
            combined = combined[:20_000]

        return combined

    def _build_prompt(
        self,
        text: str,
        num_questions: int,
        batch_index: int,
        total_batches: int,
    ) -> str:
        batch_instruction = ""
        if total_batches > 1:
            batch_instruction = f"""
This is batch {batch_index + 1} of {total_batches}.
Create different questions from the other batches.
Do not repeat concepts or questions.
"""

        return f"""
You are an expert educational assessment creator.

Generate exactly {num_questions} multiple-choice questions from the study material.

{batch_instruction}

Rules:
1. Return exactly {num_questions} questions.
2. Each question must have exactly four options: A, B, C and D.
3. Only one option must be correct.
4. Test understanding, application, analysis or calculation.
5. Do not write questions such as "According to the text".
6. Do not mention pages, documents, passages or study material.
7. Make incorrect options plausible.
8. Keep questions and options concise.
9. Keep each explanation to one short sentence.
10. Use the source reference number for source_page.
11. Return JSON only. Do not use Markdown fences.

Required JSON format:
{{
  "questions": [
    {{
      "question_text": "Question",
      "option_a": "Option A",
      "option_b": "Option B",
      "option_c": "Option C",
      "option_d": "Option D",
      "correct_answer": "A",
      "explanation": "Short explanation",
      "source_page": 1
    }}
  ]
}}

STUDY MATERIAL:
\"\"\"
{text}
\"\"\"

Return exactly {num_questions} questions now.
"""

    def _parse_response(self, content: str) -> List[dict]:
        content = content.strip()

        fenced_match = re.search(
            r"```(?:json)?\s*(.*?)```",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if fenced_match:
            content = fenced_match.group(1).strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            object_match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not object_match:
                return []
            try:
                data = json.loads(object_match.group(0))
            except json.JSONDecodeError:
                return []

        if isinstance(data, dict):
            raw_questions = data.get("questions", [])
        elif isinstance(data, list):
            raw_questions = data
        else:
            return []

        valid_questions = []

        bad_phrases = (
            "according to the text",
            "according to the passage",
            "based on the material",
            "based on the document",
            "in the passage",
            "in the text",
            "the document says",
            "the material states",
            "as mentioned in",
        )

        for item in raw_questions:
            if not isinstance(item, dict):
                continue

            question_text = str(item.get("question_text", "")).strip()
            if not question_text:
                continue

            if any(phrase in question_text.lower() for phrase in bad_phrases):
                continue

            option_a = str(item.get("option_a", "")).strip()
            option_b = str(item.get("option_b", "")).strip()
            option_c = str(item.get("option_c", "")).strip()
            option_d = str(item.get("option_d", "")).strip()

            if not all((option_a, option_b, option_c, option_d)):
                continue

            correct_answer = str(item.get("correct_answer", "")).strip().upper()
            if correct_answer not in ("A", "B", "C", "D"):
                continue

            source_page = item.get("source_page")
            try:
                source_page = int(source_page)
            except (TypeError, ValueError):
                source_page = None

            valid_questions.append(
                {
                    "question_text": question_text,
                    "option_a": option_a,
                    "option_b": option_b,
                    "option_c": option_c,
                    "option_d": option_d,
                    "correct_answer": correct_answer,
                    "explanation": str(item.get("explanation", "")).strip(),
                    "source_page": source_page,
                }
            )

        return valid_questions

    @staticmethod
    def _extract_retry_seconds(message: str) -> int:
        match = re.search(r"(\d+(?:\.\d+)?)\s*s", message.lower())
        if match:
            try:
                return max(5, int(float(match.group(1))) + 1)
            except ValueError:
                pass
        return 60

# ============================================================
# GENERATION MANAGER
# ============================================================

class ExamGenerationManager:
    @staticmethod
    def get_quota_status():
        return GeminiRateLimiter().get_status()

    @staticmethod
    def _task_key(task_id):
        return f"question_generation:task:{task_id}"

    @staticmethod
    def _review_key(review_id):
        return f"question_generation:review:{review_id}"

    @staticmethod
    def _read_count(payload: dict) -> int:
        choice = str(payload.get("question_count_choice", "10")).strip().lower()

        if choice == "custom":
            raw_count = payload.get("custom_count")
        else:
            raw_count = choice

        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            raise ValueError("Invalid question count.")

        if count < MIN_QUESTIONS or count > MAX_QUESTIONS:
            raise ValueError(
                f"Question count must be between {MIN_QUESTIONS} and {MAX_QUESTIONS}."
            )

        return count

    @classmethod
    def start_generation(cls, request, payload: dict) -> Tuple[int, dict]:
        try:
            count = cls._read_count(payload)
            plan = compute_batch_plan(count)
        except ValueError as exc:
            return 400, {"success": False, "error": str(exc)}

        doc_ids = payload.get("documents") or []

        if not isinstance(doc_ids, list) or not doc_ids:
            return 400, {"success": False, "error": "Select at least one document."}

        limiter = GeminiRateLimiter()
        can_start, message, retry_after = limiter.can_start(plan["total_batches"])

        if not can_start:
            return 429, {
                "success": False,
                "error": message,
                "retry_after": retry_after,
                "quota": limiter.get_status(),
                "plan": plan,
            }

        generator = QuestionGenerator()
        if not generator.available:
            return 503, {
                "success": False,
                "error": "Gemini is not configured. Check GEMINI_API_KEY and google-genai.",
            }

        try:
            text_by_page, source_map = cls._extract_text(request=request, payload=payload)
        except ValueError as exc:
            return 400, {"success": False, "error": str(exc)}
        except Exception:
            logger.exception("Document text extraction failed.")
            return 500, {"success": False, "error": "Could not read the selected documents."}

        if not text_by_page:
            return 400, {"success": False, "error": "No readable text was found in the selected pages."}

        task_id = str(uuid.uuid4())

        state = {
            "task_id": task_id,
            "user_id": request.user.id,
            "total_questions": plan["total_questions"],
            "batch_sizes": plan["batch_sizes"],
            "total_batches": plan["total_batches"],
            "batch_index": 0,
            "attempts": 0,
            "collected": [],
            "text_by_page": text_by_page,
            "source_map": source_map,
            "document_ids": [str(value) for value in doc_ids],
        }

        cache.set(cls._task_key(task_id), state, timeout=TASK_TIMEOUT)

        request.session["gen_task_id"] = task_id
        request.session["gen_plan"] = plan
        request.session["generation_doc_ids"] = [str(value) for value in doc_ids]

        request.session.pop("review_task_id", None)
        request.session.pop("generated_questions", None)
        request.session.modified = True

        return 200, {
            "success": True,
            "plan": plan,
            "message": f"{count} questions will be generated in {plan['total_batches']} batch(es).",
            "quota": limiter.get_status(),
        }

    @classmethod
    def process_next_batch(cls, request) -> Tuple[int, dict]:
        task_id = request.session.get("gen_task_id")

        if not task_id:
            return 400, {"success": False, "error": "No active generation task. Start again."}

        state = cache.get(cls._task_key(task_id))

        if not state:
            return 410, {"success": False, "error": "The generation task expired. Please start generation again."}

        if int(state.get("user_id")) != int(request.user.id):
            return 403, {"success": False, "error": "Permission denied."}

        processing_key = f"{cls._task_key(task_id)}:processing"
        processing_token = str(uuid.uuid4())

        if not cache.add(processing_key, processing_token, timeout=300):
            return 409, {
                "success": False,
                "retryable": True,
                "error": "A batch is already being processed.",
                "retry_after": 3,
            }

        try:
            batch_index = int(state.get("batch_index", 0))
            total_batches = int(state["total_batches"])

            if batch_index >= total_batches:
                return 200, cls._finish_generation(request=request, state=state)

            batch_size = int(state["batch_sizes"][batch_index])
            generator = QuestionGenerator()

            if not generator.available:
                return 503, {"success": False, "error": "Gemini is not configured. Check GEMINI_API_KEY."}

            estimated_tokens = (
                len(generator._prepare_text(state["text_by_page"])) // 4
                + batch_size * 150
                + 1000
            )

            limiter = GeminiRateLimiter()
            acquired, message, retry_after = limiter.acquire(estimated_tokens=estimated_tokens)

            if not acquired:
                return 429, {
                    "success": False,
                    "retryable": True,
                    "error": message,
                    "retry_after": retry_after,
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "progress": int(batch_index / total_batches * 100),
                    "quota": limiter.get_status(),
                }

            try:
                questions = generator.generate(
                    text_by_page=state["text_by_page"],
                    num_questions=batch_size,
                    batch_index=batch_index,
                    total_batches=total_batches,
                )
            except ProviderRateLimitError as exc:
                return 429, {
                    "success": False,
                    "retryable": True,
                    "error": str(exc),
                    "retry_after": exc.retry_after,
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "progress": int(batch_index / total_batches * 100),
                    "quota": limiter.get_status(),
                }
            except IncompleteBatchError as exc:
                attempts = int(state.get("attempts", 0)) + 1
                state["attempts"] = attempts

                if attempts <= 3:
                    cache.set(cls._task_key(task_id), state, timeout=TASK_TIMEOUT)
                    return 503, {
                        "success": False,
                        "retryable": True,
                        "error": f"{exc} Retrying this batch automatically.",
                        "retry_after": 5,
                        "batch_index": batch_index,
                        "total_batches": total_batches,
                        "progress": int(batch_index / total_batches * 100),
                    }

                return 502, {
                    "success": False,
                    "error": "Gemini repeatedly returned an incomplete batch. Please try again.",
                }
            except Exception as exc:
                message = str(exc).lower()
                if any(
                    phrase in message
                    for phrase in ("timeout", "temporarily", "503", "502", "connection", "internal")
                ):
                    return 503, {
                        "success": False,
                        "retryable": True,
                        "error": "Gemini temporarily failed. Retrying automatically.",
                        "retry_after": 8,
                        "batch_index": batch_index,
                        "total_batches": total_batches,
                        "progress": int(batch_index / total_batches * 100),
                    }

                logger.exception("Gemini generation failed.")
                return 500, {"success": False, "error": str(exc)}

            if len(questions) != batch_size:
                return 503, {
                    "success": False,
                    "retryable": True,
                    "error": f"Expected {batch_size} questions but received {len(questions)}.",
                    "retry_after": 5,
                }

            state["collected"].extend(questions)
            state["batch_index"] = batch_index + 1
            state["attempts"] = 0

            cache.set(cls._task_key(task_id), state, timeout=TASK_TIMEOUT)

            completed_batches = state["batch_index"]
            done = completed_batches >= total_batches

            if done:
                return 200, cls._finish_generation(request=request, state=state)

            progress = int(completed_batches / total_batches * 100)

            return 200, {
                "success": True,
                "done": False,
                "batch_index": completed_batches,
                "total_batches": total_batches,
                "batch_size": batch_size,
                "progress": progress,
                "total_so_far": len(state["collected"]),
                "total_questions": state["total_questions"],
                "quota": limiter.get_status(),
            }

        finally:
            cache.delete(processing_key)

    @classmethod
    def _extract_text(cls, request, payload: dict) -> Tuple[Dict[int, str], Dict[str, dict]]:
        doc_ids = payload.get("documents") or []
        documents = list(
            Document.objects.filter(
                id__in=doc_ids,
                uploaded_by=request.user,
                status="ready",
            )
        )

        if not documents:
            raise ValueError("No valid ready documents were selected.")

        source_type = str(payload.get("source_type", "entire")).lower()
        text_by_page = {}
        source_map = {}
        reference_number = 1

        for document in documents:
            if str(document.file_type).lower() != "pdf":
                continue

            file_path = document.file.path

            if source_type == "entire":
                pages = None
            elif source_type == "range":
                try:
                    page_from = int(payload.get("page_from") or 1)
                    page_to = int(
                        payload.get("page_to")
                        or document.page_count
                        or PDFService.get_page_count(file_path)
                    )
                except (TypeError, ValueError):
                    raise ValueError("Page range must contain valid numbers.")

                if page_from < 1 or page_to < page_from:
                    raise ValueError("The selected page range is invalid.")

                pages = PDFService.get_pages_for_range(file_path, page_from, page_to)
            elif source_type == "specific":
                pages = PDFService.parse_specific_pages(str(payload.get("specific_pages", "")))
                if not pages:
                    raise ValueError("Enter at least one valid page number.")
            elif source_type == "random":
                try:
                    random_count = int(payload.get("random_page_count") or 5)
                except (TypeError, ValueError):
                    raise ValueError("Random page count must be a valid number.")

                if random_count < 1:
                    raise ValueError("Random page count must be at least 1.")

                pages = PDFService.get_random_pages(file_path, random_count)
            else:
                raise ValueError("Invalid source page selection.")

            extracted = PDFService.extract_text_from_pages(file_path, pages)

            for original_page, page_text in extracted.items():
                page_text = str(page_text).strip()
                if not page_text:
                    continue

                text_by_page[reference_number] = (
                    f"[Document: {document.title}; original page: {original_page}]\n{page_text}"
                )
                source_map[str(reference_number)] = {
                    "document_id": str(document.id),
                    "page": int(original_page),
                }
                reference_number += 1

        return text_by_page, source_map

    @classmethod
    def _finish_generation(cls, request, state: dict) -> dict:
        questions = list(state.get("collected", []))
        source_map = state.get("source_map", {})
        first_source = next(iter(source_map.values()), None)

        for question in questions:
            source_page = question.get("source_page")
            try:
                source_key = str(int(source_page))
            except (TypeError, ValueError):
                source_key = None

            source = source_map.get(source_key) if source_key else first_source

            if source:
                question["source_document_id"] = source["document_id"]
                question["source_page"] = source["page"]
            else:
                question["source_document_id"] = None
                question["source_page"] = None

        review_id = str(uuid.uuid4())

        cache.set(
            cls._review_key(review_id),
            {
                "questions": questions,
                "document_ids": state.get("document_ids", []),
            },
            timeout=REVIEW_TIMEOUT,
        )

        request.session["review_task_id"] = review_id
        request.session["generation_doc_ids"] = state.get("document_ids", [])
        request.session.pop("generated_questions", None)

        task_id = state.get("task_id")
        if task_id:
            cache.delete(cls._task_key(task_id))

        for key in ("gen_task_id", "gen_plan"):
            request.session.pop(key, None)

        request.session.modified = True

        return {
            "success": True,
            "done": True,
            "progress": 100,
            "batch_index": state["total_batches"],
            "total_batches": state["total_batches"],
            "count": len(questions),
            "total_so_far": len(questions),
            "total_questions": state["total_questions"],
            "quota": GeminiRateLimiter().get_status(),
        }

    @classmethod
    def get_review_data(cls, request) -> dict:
        review_id = request.session.get("review_task_id")
        if review_id:
            data = cache.get(cls._review_key(review_id))
            if data:
                return data

        return {
            "questions": request.session.get("generated_questions", []),
            "document_ids": request.session.get("generation_doc_ids", []),
        }

    @classmethod
    def clear_review_data(cls, request):
        review_id = request.session.get("review_task_id")
        if review_id:
            cache.delete(cls._review_key(review_id))

        request.session.pop("review_task_id", None)
        request.session.pop("generated_questions", None)
        request.session.pop("generation_doc_ids", None)
        request.session.modified = True