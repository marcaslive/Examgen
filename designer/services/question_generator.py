# designer/services/question_generator.py

import json
import logging
import re
from typing import Dict, List, Optional

from django.conf import settings

logger = logging.getLogger(__name__)

# NEW Gemini SDK (google-genai)
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    logger.warning("google-genai not installed. Run: pip install google-genai")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class QuestionGenerator:
    """Generate intelligent MCQ questions using Gemini (free) or OpenAI."""

    def __init__(self):
        self.gemini_key = getattr(settings, 'GEMINI_API_KEY', '')
        self.openai_key = getattr(settings, 'OPENAI_API_KEY', '')
        self.model_name = getattr(settings, 'AI_MODEL', 'gemini-3.6-flash')

        self.gemini_client = None
        self.openai_client = None

        # Initialize Gemini (NEW SDK)
        if HAS_GEMINI and self.gemini_key:
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_key)
                logger.info(f"✅ Gemini client ready: {self.model_name}")
            except Exception as e:
                logger.error(f"❌ Gemini init failed: {e}")

        # Initialize OpenAI (fallback)
        if HAS_OPENAI and self.openai_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_key)
                logger.info(f"✅ OpenAI ready (fallback)")
            except Exception as e:
                logger.error(f"❌ OpenAI init failed: {e}")

    # ==========================================================
    # MAIN ENTRY POINT
    # ==========================================================
    def generate_questions(
        self,
        text_by_page: Dict[int, str],
        num_questions: int,
        source_document_id: Optional[str] = None
    ) -> List[Dict]:
        """Generate intelligent MCQ questions from PDF text."""

        if not text_by_page or num_questions <= 0:
            return []

        combined_text = self._prepare_text(text_by_page)
        if not combined_text.strip():
            return []

        # Try Gemini first
        if self.gemini_client:
            try:
                questions = self._generate_with_gemini(combined_text, num_questions, text_by_page)
                if questions:
                    logger.info(f"✅ Generated {len(questions)} questions with Gemini")
                    return questions
            except Exception as e:
                logger.exception(f"❌ Gemini failed: {e}")

        # Try OpenAI as fallback
        if self.openai_client:
            try:
                questions = self._generate_with_openai(combined_text, num_questions, text_by_page)
                if questions:
                    logger.info(f"✅ Generated {len(questions)} questions with OpenAI")
                    return questions
            except Exception as e:
                logger.exception(f"❌ OpenAI failed: {e}")

        logger.warning("⚠ All AI services failed. Using fallback.")
        return self._generate_fallback_questions(text_by_page, num_questions)

    # ==========================================================
    # TEXT PREPARATION
    # ==========================================================
    def _prepare_text(self, text_by_page: Dict[int, str]) -> str:
        parts = []
        for page_num in sorted(text_by_page.keys()):
            text = str(text_by_page[page_num]).strip()
            if not text:
                continue
            if len(text) > 3000:
                text = text[:3000] + "\n[truncated]"
            parts.append(f"[Page {page_num}]\n{text}")

        combined = "\n\n".join(parts)
        if len(combined) > 20000:
            combined = combined[:20000] + "\n\n[Document truncated]"
        return combined

    # ==========================================================
    # AI PROMPT
    # ==========================================================
    def _build_prompt(self, text: str, num_questions: int) -> str:
        return f"""You are an expert exam question creator. Generate exactly {num_questions} INTELLIGENT multiple-choice questions from the material below.

CRITICAL RULES:
1. Questions must test UNDERSTANDING, APPLICATION, ANALYSIS, or CALCULATION.
2. NEVER write: "According to the text...", "On page X...", "Based on the material...", "In the passage...", "The document says..."
3. Questions must be SELF-CONTAINED — students should understand without seeing the source.
4. Ask REAL questions like:
   - "What is the primary purpose of X?"
   - "If Y increases, what happens to Z?"
   - "Calculate the value of..." (with actual numbers)
   - "Which of the following best describes...?"
5. Each question has 4 options (A, B, C, D) — ONE correct answer.
6. Wrong answers must be PLAUSIBLE — related but incorrect.
7. Vary difficulty and topics — do NOT repeat concepts.
8. Include brief explanation.

BAD ❌: "Based on the material: 'V=IR'. Which is correct?"
GOOD ✅: "If a 12V battery connects to a 4Ω resistor, what current flows?"
       A) 2A  B) 3A  C) 4A  D) 6A  Answer: B

Return ONLY valid JSON in this format:
{{
  "questions": [
    {{
      "question_text": "The intelligent question here",
      "option_a": "First option",
      "option_b": "Second option",
      "option_c": "Third option",
      "option_d": "Fourth option",
      "correct_answer": "A",
      "explanation": "Brief explanation",
      "source_page": 1
    }}
  ]
}}

STUDY MATERIAL:
\"\"\"
{text}
\"\"\"

Generate exactly {num_questions} intelligent questions now."""

    # ==========================================================
    # GEMINI GENERATION (NEW SDK)
    # ==========================================================
    def _generate_with_gemini(self, text, num_questions, text_by_page):
        prompt = self._build_prompt(text, num_questions)

        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=65536,
                response_mime_type="application/json",
            )
        )

        content = response.text
        if not content:
            return []

        return self._parse_response(content, text_by_page)

    # ==========================================================
    # OPENAI GENERATION
    # ==========================================================
    def _generate_with_openai(self, text, num_questions, text_by_page):
        prompt = self._build_prompt(text, num_questions)

        response = self.openai_client.chat.completions.create(
            model=self.model_name if 'gpt' in self.model_name else 'gpt-4o-mini',
            messages=[
                {"role": "system", "content": "You are an expert exam creator. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            return []

        return self._parse_response(content, text_by_page)

    # ==========================================================
    # PARSE + VALIDATE RESPONSE
    # ==========================================================
    def _parse_response(self, content: str, text_by_page: Dict[int, str]) -> List[Dict]:
        content = content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
            content = re.sub(r"\s*```$", "", content).strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                logger.error(f"Cannot parse JSON: {content[:500]}")
                return []
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []

        if isinstance(data, list):
            questions_raw = data
        elif isinstance(data, dict):
            questions_raw = data.get('questions', [])
        else:
            return []

        # Reject questions that reference source text
        bad_phrases = [
            'according to the text', 'according to the passage',
            'on page', 'in the passage', 'in the text',
            'the author states', 'the passage says',
            'based on the reading', 'based on the material',
            'the document says', 'the material states',
            'as mentioned in', 'as stated in',
        ]

        valid_questions = []
        page_numbers = sorted(text_by_page.keys())

        for q in questions_raw:
            if not isinstance(q, dict):
                continue

            question_text = str(q.get("question_text", "")).strip()
            if not question_text:
                continue

            q_lower = question_text.lower()
            if any(phrase in q_lower for phrase in bad_phrases):
                logger.warning(f"Rejecting: {question_text[:60]}...")
                continue

            option_a = str(q.get("option_a", "")).strip()
            option_b = str(q.get("option_b", "")).strip()
            option_c = str(q.get("option_c", "")).strip()
            option_d = str(q.get("option_d", "")).strip()

            if not all([option_a, option_b, option_c, option_d]):
                continue

            correct = str(q.get("correct_answer", "")).upper().strip()
            if correct not in ["A", "B", "C", "D"]:
                continue

            source_page = q.get("source_page")
            try:
                source_page = int(source_page)
                if source_page not in page_numbers:
                    source_page = page_numbers[0] if page_numbers else None
            except (ValueError, TypeError):
                source_page = page_numbers[0] if page_numbers else None

            valid_questions.append({
                "question_text": question_text,
                "option_a": option_a,
                "option_b": option_b,
                "option_c": option_c,
                "option_d": option_d,
                "correct_answer": correct,
                "explanation": str(q.get("explanation", "")).strip(),
                "source_page": source_page,
            })

        return valid_questions

    # ==========================================================
    # FALLBACK (only if all AI fails)
    # ==========================================================
    def _generate_fallback_questions(self, text_by_page, num_questions):
        logger.warning("⚠ Using placeholder - configure GEMINI_API_KEY properly")
        first_page = min(text_by_page.keys()) if text_by_page else 1

        return [{
            "question_text": "⚠ AI service unavailable - check GEMINI_API_KEY in settings.py",
            "option_a": "Install google-genai",
            "option_b": "Verify API key",
            "option_c": "Both A and B",
            "option_d": "Neither",
            "correct_answer": "C",
            "explanation": "Run: pip install google-genai. Then verify your GEMINI_API_KEY is correct.",
            "source_page": first_page,
        }]