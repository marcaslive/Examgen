# designer/services/question_generator.py

import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from django.conf import settings
from django.core.cache import cache

from designer.models import Document
from designer.services.pdf_service import PDFService

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================
# 1. RATE LIMITER (Protects RPM/RPD)
# ============================================================
class GeminiRateLimiter:
    RPM_LIMIT = 5
    RPD_LIMIT = 20
    MIN_INTERVAL_SEC = 12
    CACHE_PREFIX = "gemini_rl:"

    def _day_key(self): return f"{self.CACHE_PREFIX}rpd:{datetime.utcnow().strftime('%Y%m%d')}"
    def _min_key(self): return f"{self.CACHE_PREFIX}rpm:{datetime.utcnow().strftime('%Y%m%d%H%M')}"
    def _last_key(self): return f"{self.CACHE_PREFIX}last_call"

    def get_status(self) -> dict:
        rpd = cache.get(self._day_key(), 0)
        rpm = cache.get(self._min_key(), 0)
        last = cache.get(self._last_key())

        wait = 0
        if last:
            elapsed = time.time() - float(last)
            if elapsed < self.MIN_INTERVAL_SEC:
                wait = int(self.MIN_INTERVAL_SEC - elapsed) + 1

        now = datetime.utcnow()
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        reset_sec = int((tomorrow - now).total_seconds())

        return {
            "rpd_remaining": max(0, self.RPD_LIMIT - rpd),
            "rpd_limit": self.RPD_LIMIT,
            "rpm_remaining": max(0, self.RPM_LIMIT - rpm),
            "rpm_limit": self.RPM_LIMIT,
            "wait_seconds": wait,
            "can_request": (rpd < self.RPD_LIMIT and rpm < self.RPM_LIMIT and wait == 0),
        }

    def can_afford(self, batches_needed: int) -> Tuple[bool, str, int]:
        st = self.get_status()
        if st["rpd_remaining"] < batches_needed:
            return False, f"Daily limit too low. Need {batches_needed}, have {st['rpd_remaining']}.", 3600
        if st["wait_seconds"] > 0:
            return False, f"Cooldown active. Wait {st['wait_seconds']}s.", st["wait_seconds"]
        if st["rpm_remaining"] <= 0:
            return False, "Minute limit reached.", 60
        return True, "", 0

    def record_call(self):
        cache.set(self._day_key(), cache.get(self._day_key(), 0) + 1, 86400)
        cache.set(self._min_key(), cache.get(self._min_key(), 0) + 1, 120)
        cache.set(self._last_key(), str(time.time()), 120)


# ============================================================
# 2. CORE AI GENERATOR
# ============================================================
class QuestionGenerator:
    def __init__(self):
        self.gemini_key = getattr(settings, "GEMINI_API_KEY", "")
        self.model_name = getattr(settings, "AI_MODEL", "gemini-2.0-flash")
        self.gemini_client = genai.Client(api_key=self.gemini_key) if HAS_GEMINI and self.gemini_key else None

    def generate(self, text: str, num_questions: int, batch_index: int, total_batches: int) -> list:
        if not self.gemini_client:
            raise Exception("Gemini client not configured. Check GEMINI_API_KEY.")

        prompt = f"""You are an expert exam question creator. Generate exactly {num_questions} INTELLIGENT multiple-choice questions from the material below.
{"Cover DIFFERENT topics than previous batches. Do NOT repeat concepts." if total_batches > 1 else ""}
CRITICAL RULES:
1. Questions must test UNDERSTANDING or APPLICATION.
2. NEVER write: "According to the text..."
3. Must be SELF-CONTAINED.
4. Each question has 4 options (A, B, C, D) — ONE correct answer.
Return ONLY valid JSON in this format:
{{"questions": [{{"question_text": "...", "option_a": "...", "option_b": "...", "option_c": "...", "option_d": "...", "correct_answer": "A", "explanation": "..."}}]}}

STUDY MATERIAL:
\"\"\"{text[:25000]}\"\"\""""

        max_out = min(8192, max(2048, num_questions * 400))
        res = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=max_out, response_mime_type="application/json")
        )
        return self._parse(res.text)

    def _parse(self, content: str) -> list:
        try:
            content = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
            content = re.sub(r"\s*```$", "", content).strip()
            data = json.loads(content)
            qs = data.get("questions", []) if isinstance(data, dict) else data
            
            # Ensure proper format
            valid_qs = []
            for q in qs:
                if all(k in q for k in ("question_text", "option_a", "option_b", "option_c", "option_d", "correct_answer")):
                    valid_qs.append(q)
            return valid_qs
        except Exception:
            return []


# ============================================================
# 3. MANAGER (The Orchestrator)
# ============================================================
class ExamGenerationManager:
    @staticmethod
    def get_quota_status() -> dict:
        return GeminiRateLimiter().get_status()

    @classmethod
    def start_generation(cls, request, payload: dict) -> Tuple[int, dict]:
        try:
            count = int(payload.get("custom_count") or payload.get("question_count_choice", 10))
        except ValueError:
            return 400, {"success": False, "error": "Invalid question count."}
        
        count = max(10, min(500, count))
        batches = [count] if count <= 50 else [20] * (count // 20) + ([count % 20] if count % 20 else [])
        
        limiter = GeminiRateLimiter()
        ok, msg, retry = limiter.can_afford(len(batches))
        if not ok:
            return 429, {"success": False, "error": msg, "retry_after": retry}

        # Extracted text via your PDFService
        text_by_page = cls._extract_text(payload)
        if not text_by_page.strip():
            return 400, {"success": False, "error": "Could not extract readable text from the selected documents."}

        task_id = str(uuid.uuid4())
        cache.set(f"gen_text_{task_id}", text_by_page, timeout=3600)
        
        request.session["gen_task_id"] = task_id
        request.session["gen_plan"] = {"sizes": batches, "total": len(batches)}
        request.session["gen_collected"] = []
        request.session["gen_index"] = 0
        request.session.modified = True

        return 200, {
            "success": True,
            "plan": {"total_questions": count, "total_batches": len(batches), "estimated_minutes": round(len(batches)*15/60, 1)},
            "quota": limiter.get_status()
        }

    @classmethod
    def process_next_batch(cls, request) -> Tuple[int, dict]:
        task_id = request.session.get("gen_task_id")
        plan = request.session.get("gen_plan")
        idx = request.session.get("gen_index", 0)

        if not task_id or not plan:
            return 400, {"success": False, "error": "Session lost. Start again."}

        if idx >= plan["total"]:
            return 200, cls._finish_generation(request)

        text_by_page = cache.get(f"gen_text_{task_id}", "")
        if not text_by_page:
            return 500, {"success": False, "error": "Text expired from memory."}

        limiter = GeminiRateLimiter()
        ok, msg, retry = limiter.can_afford(1)
        if not ok:
            return 429, {"success": False, "error": msg, "retry_after": retry}

        size = plan["sizes"][idx]
        ai = QuestionGenerator()
        limiter.record_call()
        
        try:
            questions = ai.generate(str(text_by_page), size, idx, plan["total"])
        except Exception as e:
            if "429" in str(e):
                return 429, {"success": False, "error": "Rate limit hit mid-run.", "retry_after": 60}
            return 500, {"success": False, "error": str(e)}

        collected = request.session.get("gen_collected", [])
        collected.extend(questions)
        request.session["gen_collected"] = collected
        request.session["gen_index"] = idx + 1
        request.session.modified = True

        is_done = (idx + 1) >= plan["total"]
        if is_done:
            return 200, cls._finish_generation(request)

        return 200, {
            "success": True,
            "done": False,
            "batch_index": idx + 1,
            "total_batches": plan["total"],
            "progress": int((idx + 1) / plan["total"] * 100),
            "total_so_far": len(collected),
            "quota": limiter.get_status()
        }

    @classmethod
    def _extract_text(cls, payload: dict) -> str:
        """Pulls exact pages based on UI selection using your PDFService."""
        doc_ids = payload.get("documents", [])
        source_type = payload.get("source_type", "entire")
        combined_text = ""

        docs = Document.objects.filter(id__in=doc_ids)
        for doc in docs:
            file_path = doc.file.path
            pages = None  # None means all pages in PDFService
            
            if source_type == "range":
                start = int(payload.get("page_from", 1))
                end = int(payload.get("page_to", 10))
                pages = PDFService.get_pages_for_range(file_path, start, end)
            elif source_type == "specific":
                pages = PDFService.parse_specific_pages(payload.get("specific_pages", ""))
            elif source_type == "random":
                count = int(payload.get("random_page_count", 10))
                pages = PDFService.get_random_pages(file_path, count)

            # extract text via your PDFService
            text_dict = PDFService.extract_text_from_pages(file_path, pages)
            for p_num, p_text in sorted(text_dict.items()):
                combined_text += f"\n[Doc: {doc.title} | Page: {p_num}]\n{p_text}\n"

        return combined_text

    @classmethod
    def _finish_generation(cls, request) -> dict:
        collected = request.session.get("gen_collected", [])
        
        # Save exact array to session for the review page to pick up!
        request.session["review_questions"] = collected
        
        task_id = request.session.get("gen_task_id")
        if task_id: cache.delete(f"gen_text_{task_id}")
        
        for k in ["gen_task_id", "gen_plan", "gen_collected", "gen_index"]:
            request.session.pop(k, None)
        request.session.modified = True

        return {
            "success": True,
            "done": True,
            "progress": 100,
            "count": len(collected),
            "quota": GeminiRateLimiter().get_status()
        }