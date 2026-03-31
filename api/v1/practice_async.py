"""
Async Practice API for SkillBot
Practice session management endpoints with analytics
"""

import logging
import aiofiles
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from bson import ObjectId

from fastapi import APIRouter, Request, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field, validator, root_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.database import DatabaseManager
from core.cache import CacheManager
from api.v1.auth_async import get_current_user, get_database, get_cache
from config_async import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Gate bridge (SWM-011)
#
# All LLM calls — text-only AND vision/multimodal — MUST be routed through
# the shared gate (C4) with caller_id ``pcr_practice``.  If the gate module
# is available, it is the exclusive path — there is NO silent fallback to a
# direct provider.
#
# Vision/multimodal calls use ``gate.call(messages=...)`` which forwards the
# pre-built messages array (text + image_url parts) to the provider.
#
# If the gate module is not importable (exam-conductor not deployed),
# RuntimeError is raised — there is NO fallback to direct providers.
# The deployment must include exam-conductor for practice to function.
# ---------------------------------------------------------------------------

_gate_module = None          # lazily imported exam-conductor.llm_gate
_gate_import_attempted = False  # True once we have tried (success or failure)
_gate_unavailable = False    # True when import was attempted and failed


def _try_load_gate_module():
    """Attempt to import the LLM gate module once.  Returns the module or None."""
    global _gate_module, _gate_import_attempted, _gate_unavailable
    if _gate_module is not None:
        return _gate_module
    if _gate_import_attempted:
        return None
    _gate_import_attempted = True
    try:
        from api.v1._exampen_imports import load_exampen
        _gate_module = load_exampen("llm_gate")
        logger.info("LLM gate module loaded for practice bridge (SWM-011)")
        return _gate_module
    except ImportError:
        _gate_unavailable = True
        # CRITICAL: C4 requires all LLM calls through the gate.  If the gate
        # is not importable the deployment is non-compliant.
        logger.critical(
            "SWM-011 C4 VIOLATION: exam-conductor.llm_gate not importable — "
            "practice LLM calls (text + vision) will bypass the gate.  "
            "Deploy exam-conductor to restore compliance."
        )
        return None


async def _gate_text_call(
    db: DatabaseManager,
    current_user: Dict[str, Any],
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    model_override: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Route a text-only LLM call through the shared gate.

    Returns a dict shaped like ``AsyncOpenAIService.chat_completion_async``
    output (``success``, ``response``, etc.) or ``None`` **only** when the
    gate module was never importable (deployment issue).

    If the gate *is* loaded but the call fails at runtime, the exception
    propagates — there is no silent fallback to a direct provider.
    """
    gate_mod = _try_load_gate_module()
    if gate_mod is None:
        raise RuntimeError(
            "SWM-011 C4: LLM gate not available — exam-conductor not deployed. "
            "All LLM calls must go through the shared gate (C4)."
        )

    db_name = current_user.get("db_name")
    if not db_name:
        raise RuntimeError("SWM-011: No db_name in user context — cannot route through gate")

    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise RuntimeError(f"SWM-011: Could not obtain tenant DB '{db_name}' — cannot route through gate")

    gate = gate_mod.LLMGate(tenant_db)
    await gate.initialize()

    # Build the full prompt (system + user) since the gate takes a single prompt string
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"

    from config_async import OPENAI_MODEL
    model_id = model_override or OPENAI_MODEL

    gate_resp = await gate.call(
        model_id=model_id,
        prompt=full_prompt,
        caller_id="pcr_practice",
        max_output_tokens=max_tokens,
        temperature=temperature,
    )

    # Map GateResponse -> dict matching AsyncOpenAIService output shape
    return {
        "success": True,
        "response": gate_resp.content,
        "model": gate_resp.usage.model,
        "usage": {
            "prompt_tokens": gate_resp.usage.input_tokens,
            "completion_tokens": gate_resp.usage.output_tokens,
            "total_tokens": gate_resp.usage.total_tokens,
        },
    }

async def _gate_vision_call(
    db: DatabaseManager,
    current_user: Dict[str, Any],
    images: List[str],
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.3,
    model_override: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Route a vision/multimodal LLM call through the shared gate.

    Builds an OpenAI-style messages array with text + image_url parts and
    forwards it via ``gate.call(messages=...)``.

    Returns a dict shaped like ``AsyncOpenAIService.analyze_images_and_text_async``
    output (``success``, ``response``, etc.) or ``None`` **only** when the gate
    module was never importable (deployment issue).

    If the gate *is* loaded but the call fails at runtime, the exception
    propagates — there is no silent fallback to a direct provider.
    """
    gate_mod = _try_load_gate_module()
    if gate_mod is None:
        raise RuntimeError(
            "SWM-011 C4: LLM gate not available — exam-conductor not deployed. "
            "All LLM calls (including vision) must go through the shared gate (C4)."
        )

    db_name = current_user.get("db_name")
    if not db_name:
        raise RuntimeError("SWM-011: No db_name in user context — cannot route through gate")

    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise RuntimeError(f"SWM-011: Could not obtain tenant DB '{db_name}' — cannot route through gate")

    gate = gate_mod.LLMGate(tenant_db)
    await gate.initialize()

    # Build multimodal messages array (OpenAI format)
    content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images or []:
        if not img:
            continue
        # Ensure data URI format
        if img.startswith("data:"):
            url = img
        else:
            url = f"data:image/png;base64,{img}"
        content_parts.append({"type": "image_url", "image_url": {"url": url}})

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content_parts})

    from config_async import OPENAI_MODEL
    model_id = model_override or OPENAI_MODEL

    gate_resp = await gate.call(
        model_id=model_id,
        prompt=prompt,
        caller_id="pcr_practice",
        messages=messages,
        max_output_tokens=max_tokens,
        temperature=temperature,
    )

    # Map GateResponse -> dict matching AsyncOpenAIService output shape
    return {
        "success": True,
        "response": gate_resp.content,
        "model": gate_resp.usage.model,
        "usage": {
            "prompt_tokens": gate_resp.usage.input_tokens,
            "completion_tokens": gate_resp.usage.output_tokens,
            "total_tokens": gate_resp.usage.total_tokens,
        },
    }


# Language detection utility for multilingual support
def detect_language(text: str) -> str:
    """
    Detect the primary language of the given text.
    
    Returns:
        'hindi' if the text contains significant Hindi (Devanagari) characters
        'english' otherwise (default)
    
    This is used to ensure LLM responses match the question/input language.
    """
    if not text:
        return 'english'
    
    # Count Devanagari characters (Hindi script range: U+0900 to U+097F)
    devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    total_alpha_count = sum(1 for char in text if char.isalpha())
    
    if total_alpha_count == 0:
        return 'english'
    
    # If more than 20% of alphabetic characters are Devanagari, consider it Hindi
    hindi_ratio = devanagari_count / total_alpha_count
    
    if hindi_ratio > 0.2:
        return 'hindi'
    
    return 'english'


def get_language_instruction(detected_language: str) -> str:
    """
    Get the language instruction for LLM prompts based on detected language.
    
    This ensures the LLM responds in the same language as the question/input.
    """
    if detected_language == 'hindi':
        return (
            "\n\n🌐 भाषा निर्देश (LANGUAGE INSTRUCTION):\n"
            "यह प्रश्न हिंदी में है। कृपया अपना पूरा उत्तर केवल हिंदी में दें।\n"
            "The question is in Hindi. You MUST respond ENTIRELY in Hindi.\n"
            "All feedback, explanations, and solutions should be in Hindi only.\n"
        )
    else:
        return (
            "\n\n🌐 LANGUAGE INSTRUCTION:\n"
            "The question is in English. Respond entirely in English.\n"
        )


def robust_json_parse(raw_response: str) -> Optional[Dict[str, Any]]:
    """
    Robustly parse JSON from LLM response, handling:
    - Markdown code blocks
    - LaTeX content with escaped backslashes
    - Incomplete/truncated JSON
    - Extra text before/after JSON
    
    Returns parsed dict or None if parsing fails.
    """
    import json
    import re
    import ast
    
    if not raw_response:
        return None
    
    # Step 1: Remove markdown code blocks
    clean = raw_response.strip()
    
    # Remove ```json ... ``` blocks
    if "```json" in clean:
        match = re.search(r'```json\s*([\s\S]*?)\s*```', clean)
        if match:
            clean = match.group(1).strip()
        else:
            clean = re.sub(r'```json\s*', '', clean)
            clean = re.sub(r'\s*```', '', clean)
    elif "```" in clean:
        match = re.search(r'```\s*([\s\S]*?)\s*```', clean)
        if match:
            clean = match.group(1).strip()
        else:
            clean = re.sub(r'```\s*', '', clean)
    
    # Step 2: Try direct JSON parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    
    # Step 3: Find JSON object boundaries (handle nested braces)
    json_str = None
    
    # Find the first { and try to match balanced braces
    start_idx = clean.find('{')
    if start_idx != -1:
        brace_count = 0
        end_idx = start_idx
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(clean)):
            char = clean[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
        
        if brace_count == 0 and end_idx > start_idx:
            json_str = clean[start_idx:end_idx + 1]
    
    if json_str:
        # Try parsing the extracted JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Step 4: Fix common JSON issues
        fixed = json_str
        
        # Fix unescaped control characters
        fixed = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', fixed)
        
        # Try again after fixing
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Step 5: Try ast.literal_eval (handles single quotes)
        try:
            result = ast.literal_eval(fixed)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass
    
    # Step 6: Last resort - regex extraction for key fields
    # Try to extract individual fields if full parse fails
    try:
        extracted = {}

        # Helper: extract a JSON string value by key name.
        # Handles escaped quotes inside the value and stops at the
        # first unescaped closing quote.
        def _extract_string_field(key: str) -> Optional[str]:
            m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', clean, re.DOTALL)
            if m:
                return m.group(1).replace('\\"', '"').replace('\\n', '\n')
            return None

        # Extract is_correct (boolean)
        is_correct_match = re.search(r'"is_correct"\s*:\s*(true|false)', clean, re.IGNORECASE)
        if is_correct_match:
            extracted["is_correct"] = is_correct_match.group(1).lower() == "true"

        # Extract score (number)
        score_match = re.search(r'"score"\s*:\s*([0-9.]+)', clean)
        if score_match:
            extracted["score"] = float(score_match.group(1))

        # Extract all string fields that the evaluation prompt requests
        _string_fields = [
            "extracted_answer", "solved_answer", "feedback", "reasoning",
            "work_shown", "what_went_wrong", "correct_solution",
        ]
        for field_name in _string_fields:
            val = _extract_string_field(field_name)
            if val:
                extracted[field_name] = val

        # If we got at least is_correct, use the extracted data
        if "is_correct" in extracted:
            logger.info(f"📊 Partial JSON extraction succeeded: fields={list(extracted.keys())}")
            return extracted

    except Exception as e:
        logger.warning(f"Partial extraction failed: {e}")

    return None


def _truncate_for_prompt(text: str, max_chars: int) -> str:
    """Trim large text blocks for prompt safety while preserving useful content."""
    if not text:
        return ""
    clean = str(text).strip()
    if len(clean) <= max_chars:
        return clean
    omitted = len(clean) - max_chars
    return f"{clean[:max_chars]}\n...[truncated {omitted} chars]"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Pydantic models
class PracticeSession(BaseModel):
    id: Optional[str] = None
    student_id: str
    mode: str = Field(..., pattern="^(practice|exam|timed)$")
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    questions_attempted: int = Field(default=0, ge=0)
    correct_answers: int = Field(default=0, ge=0)
    total_time_spent: int = Field(default=0, ge=0)  # in seconds
    started_at: datetime
    completed_at: Optional[datetime] = None
    is_completed: bool = False

class SessionQuestion(BaseModel):
    question_id: str
    answer: str
    is_correct: bool
    time_spent: int = Field(ge=0)  # in seconds
    answered_at: datetime

class SessionAnswer(BaseModel):
    question_id: str
    answer: str
    time_spent: int = Field(default=0, ge=0)

class StartSessionRequest(BaseModel):
    mode: str = Field(..., pattern="^(practice|exam|timed)$")
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    time_limit: Optional[int] = Field(None, ge=1)  # in minutes
    document_id: Optional[str] = None  # Practice set document ID

class SessionResponse(BaseModel):
    id: str
    mode: str
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    questions_attempted: int
    correct_answers: int
    accuracy_rate: float
    total_time_spent: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    is_completed: bool

class SessionsListResponse(BaseModel):
    sessions: List[SessionResponse]
    total: int
    page: int
    limit: int

class PracticeStats(BaseModel):
    total_sessions: int
    total_time_spent: int
    average_accuracy: float
    sessions_by_mode: Dict[str, int]
    recent_activity: List[Dict[str, Any]]

# ----------------------
# Helper utilities (local)
# ----------------------

async def _extract_text_from_document(base64_data: str, doc_type: str, filename: str) -> str:
    """Extract text content from uploaded PDF or DOCX document.
    
    Args:
        base64_data: Base64 encoded document data (may include data URL prefix)
        doc_type: Type of document ('pdf' or 'docx')
        filename: Original filename for logging
        
    Returns:
        Extracted text content from the document
    """
    import base64
    import io
    import tempfile
    import os
    
    try:
        # Remove data URL prefix if present
        if ',' in base64_data:
            base64_data = base64_data.split(',')[-1]
        
        # Decode base64 to bytes
        doc_bytes = base64.b64decode(base64_data)
        
        if doc_type == 'pdf':
            try:
                from pypdf import PdfReader
                pdf_file = io.BytesIO(doc_bytes)
                pdf_reader = PdfReader(pdf_file)
                
                text_content = []
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                
                return "\n\n".join(text_content)
            except ImportError:
                logger.error("pypdf not installed. Cannot extract PDF text. Install with: pip install pypdf")
                return ""
        
        elif doc_type == 'docx':
            try:
                from docx import Document
                docx_file = io.BytesIO(doc_bytes)
                doc = Document(docx_file)
                
                text_content = []
                for para in doc.paragraphs:
                    if para.text.strip():
                        text_content.append(para.text)
                
                # Also extract text from tables
                for table in doc.tables:
                    for row in table.rows:
                        row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_text:
                            text_content.append(" | ".join(row_text))
                
                return "\n".join(text_content)
            except ImportError:
                logger.error("python-docx not installed. Cannot extract DOCX text.")
                return ""
        
        else:
            logger.warning(f"Unsupported document type: {doc_type}")
            return ""
            
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {e}")
        return ""


async def _load_question_doc(db: DatabaseManager, qid: str, is_b2c: bool = False) -> Dict[str, Any]:
    """Fetch question from MongoDB.
    
    For B2C users, uses B2C database instead of main database.
    """
    if is_b2c:
        return await db.b2c_find_one("questions", {"id": qid}) or {}
    return await db.mongo_find_one("questions", {"id": qid}) or {}


def _options_text_from_question(q: Dict[str, Any]) -> str:
    opts = q.get("options", []) or []
    if opts:
        return "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(opts)])
    enh = q.get("enhancedOptions") or []
    if enh:
        parts = []
        for i, opt in enumerate(enh):
            label = chr(65 + i)
            if isinstance(opt, dict):
                if opt.get("type") == "text" and opt.get("content"):
                    parts.append(f"{label}. {opt.get('content')}")
                elif opt.get("type") == "image":
                    desc = opt.get("description") or "image option"
                    parts.append(f"{label}. [IMAGE] {desc}")
            else:
                parts.append(f"{label}. {str(opt)}")
        return "\n".join(parts)
    return ""


def _parse_number(s: str):
    """Try to parse a string as a number. Returns float or None."""
    if not s:
        return None
    s = s.strip().replace(',', '')
    # Handle fractions like "1/2"
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            try:
                return float(parts[0].strip()) / float(parts[1].strip())
            except (ValueError, ZeroDivisionError):
                return None
    try:
        return float(s)
    except ValueError:
        return None


def _answers_are_equivalent(student_answer: str, correct_answer: str) -> bool:
    """
    Check if two answers are semantically equivalent.
    Handles: numeric (9 vs nine), case, whitespace, units, fractions.
    """
    import re as _re_equiv

    if not student_answer or not correct_answer:
        return False

    s = student_answer.strip().lower()
    c = correct_answer.strip().lower()

    # Direct match (case-insensitive)
    if s == c:
        return True

    # Word-to-number mapping
    _word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
        'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
        'half': 0.5, 'quarter': 0.25, 'third': 1/3,
    }

    # Try direct numeric parse
    s_num = _parse_number(s)
    c_num = _parse_number(c)
    if s_num is not None and c_num is not None:
        return abs(s_num - c_num) < 1e-6

    # Try word-to-number for student answer vs numeric correct answer
    s_word_num = _word_to_num.get(s)
    if s_word_num is not None and c_num is not None:
        return abs(s_word_num - c_num) < 1e-6

    # Try numeric student answer vs word correct answer
    c_word_num = _word_to_num.get(c)
    if c_word_num is not None and s_num is not None:
        return abs(c_word_num - s_num) < 1e-6

    # Both are words
    if s_word_num is not None and c_word_num is not None:
        return abs(s_word_num - c_word_num) < 1e-6

    # Strip common units and compare core value
    _unit_pattern = r'\s*(days?|hours?|minutes?|mins?|seconds?|secs?|years?|months?|weeks?|meters?|metres?|centimeters?|centimetres?|cm|mm|km|m|kg|grams?|g|mg|ml|litres?|liters?|l|%|percent|rs\.?|rupees?|₹|\$|€|£|°[cfCF]?|degrees?)\.?\s*$'
    s_stripped = _re_equiv.sub(_unit_pattern, '', s, flags=_re_equiv.IGNORECASE).strip()
    c_stripped = _re_equiv.sub(_unit_pattern, '', c, flags=_re_equiv.IGNORECASE).strip()

    if s_stripped and c_stripped and s_stripped == c_stripped:
        return True

    # Try numeric comparison after stripping units
    s_num2 = _parse_number(s_stripped)
    c_num2 = _parse_number(c_stripped)
    if s_num2 is not None and c_num2 is not None:
        return abs(s_num2 - c_num2) < 1e-6

    # Word-to-number after stripping units
    s_word2 = _word_to_num.get(s_stripped)
    if s_word2 is not None and c_num2 is not None:
        return abs(s_word2 - c_num2) < 1e-6
    c_word2 = _word_to_num.get(c_stripped)
    if c_word2 is not None and s_num2 is not None:
        return abs(c_word2 - s_num2) < 1e-6

    return False


def _resolve_correct_answer(correct_answer: str, question_doc: dict) -> dict:
    """
    Resolve the correct answer into multiple representations for robust evaluation.

    Returns dict with:
        - raw: original stored value (e.g. "A")
        - resolved_value: actual content of the option (e.g. "7" for option A)
        - display: human-readable for summary (e.g. "A (7)")
        - is_option_letter: True if the answer is A/B/C/D
    """
    result = {
        "raw": correct_answer,
        "resolved_value": correct_answer,
        "display": correct_answer,
        "is_option_letter": False,
    }

    if not correct_answer:
        return result

    ca_upper = correct_answer.strip().upper()

    # Check if it's an option letter (A-J covers most MCQ ranges)
    if len(ca_upper) == 1 and ca_upper in "ABCDEFGHIJ":
        option_index = ord(ca_upper) - ord('A')

        # Try to resolve from enhancedOptions first, then plain options
        enhanced = question_doc.get("enhancedOptions") or []
        opts = question_doc.get("options", []) or []

        resolved_content = None
        if enhanced and option_index < len(enhanced):
            opt = enhanced[option_index]
            if isinstance(opt, dict):
                content = opt.get("content", "")
                if content and content.strip():
                    resolved_content = content.strip()
        elif opts and option_index < len(opts):
            if opts[option_index] and str(opts[option_index]).strip():
                resolved_content = str(opts[option_index]).strip()

        if resolved_content:
            result["is_option_letter"] = True
            result["resolved_value"] = resolved_content
            result["display"] = f"{ca_upper} ({resolved_content})"
        else:
            # It looks like a letter but we couldn't resolve it — still mark as option letter
            result["is_option_letter"] = True

    return result


async def _figure_images_base64(q: Dict[str, Any], db: DatabaseManager = None, is_b2c: bool = False) -> List[str]:
    """Extract base64 image data for question figures.
    
    ENHANCED: Now loads images from disk/database if base64Data is not embedded.
    This ensures question diagram images are always available for LLM evaluation.
    
    Args:
        q: Question document
        db: Database manager for loading images from disk (optional)
        is_b2c: Whether this is a B2C user (uses B2C database)
        
    Returns:
        List of base64 data URLs for question figures
    """
    import os
    import base64 as base64_module
    
    imgs: List[str] = []
    
    # Helper function to load image from database/disk
    async def load_image_by_id(img_id: str) -> Optional[str]:
        if not img_id or not db:
            return None
        try:
            if is_b2c:
                img_doc = await db.b2c_find_one("images", {"_id": img_id})
            else:
                img_doc = await db.mongo_find_one("images", {"_id": img_id})
            
            if img_doc:
                if img_doc.get("base64Data"):
                    b64 = img_doc["base64Data"]
                    logger.info(f"Loaded base64Data from database for image {img_id}")
                    return b64
                elif img_doc.get("file_path"):
                    file_path = img_doc["file_path"]
                    if os.path.exists(file_path):
                        async with aiofiles.open(file_path, "rb") as f:
                            image_bytes = await f.read()
                        base64_encoded = base64_module.b64encode(image_bytes).decode('utf-8')
                        content_type = img_doc.get("content_type", "image/jpeg")
                        if not content_type.startswith("image/"):
                            content_type = "image/jpeg"
                        b64 = f"data:{content_type};base64,{base64_encoded}"
                        logger.info(f"Loaded image from disk for {img_id}: {len(b64)} bytes")
                        return b64
        except Exception as e:
            logger.error(f"Failed to load image {img_id}: {e}")
        return None
    
    # Helper to normalize base64 format
    def normalize_b64(b64: str) -> str:
        if b64 and not b64.startswith("data:image"):
            return f"data:image/png;base64,{b64}"
        return b64
    
    # 1. Extract from question_figures array
    for fig_ref in (q.get("question_figures", []) or []):
        try:
            b64 = None
            fig_id = None
            
            if isinstance(fig_ref, dict):
                b64 = fig_ref.get("base64Data")
                fig_id = fig_ref.get("id")
            elif isinstance(fig_ref, str):
                fig_id = fig_ref
            
            if not b64 and fig_id:
                b64 = await load_image_by_id(fig_id)
            
            if b64:
                imgs.append(normalize_b64(b64))
                
        except Exception as e:
            logger.warning(f"Error processing figure: {e}")
    
    # 2. Extract from images array (diagrams and other embedded images)
    for img_ref in (q.get("images", []) or []):
        try:
            b64 = None
            img_id = None
            
            if isinstance(img_ref, dict):
                # Include ALL image types, not just diagrams
                b64 = img_ref.get("base64Data")
                img_id = img_ref.get("id")
            elif isinstance(img_ref, str):
                img_id = img_ref
            
            if not b64 and img_id:
                b64 = await load_image_by_id(img_id)
            
            if b64:
                imgs.append(normalize_b64(b64))
        except Exception:
            pass
    
    # 3. Check for inline figure in the question text (base64 embedded)
    if q.get("figure") and isinstance(q.get("figure"), str):
        fig = q["figure"]
        if fig.startswith("data:image") or len(fig) > 100:  # Likely base64
            imgs.append(normalize_b64(fig))
    
    # 4. Check for questionImage field
    if q.get("questionImage"):
        qimg = q["questionImage"]
        if isinstance(qimg, str):
            if qimg.startswith("data:image") or len(qimg) > 100:
                imgs.append(normalize_b64(qimg))
            else:
                # It might be an ID
                loaded = await load_image_by_id(qimg)
                if loaded:
                    imgs.append(normalize_b64(loaded))
        elif isinstance(qimg, dict):
            b64 = qimg.get("base64Data")
            if not b64:
                b64 = await load_image_by_id(qimg.get("id"))
            if b64:
                imgs.append(normalize_b64(b64))
    
    logger.info(f"📷 Extracted {len(imgs)} question figure/diagram images for LLM evaluation")
    return imgs


async def _option_images_base64(q: Dict[str, Any], db: DatabaseManager = None, is_b2c: bool = False) -> List[Dict[str, str]]:
    """Extract images from MCQ options.
    
    Returns list of dicts with 'option' (A, B, C, D) and 'image' (base64 data URL)
    """
    import os
    import base64 as base64_module
    
    option_images: List[Dict[str, str]] = []
    
    # Helper to load image by ID
    async def load_image_by_id(img_id: str) -> Optional[str]:
        if not img_id or not db:
            return None
        try:
            if is_b2c:
                img_doc = await db.b2c_find_one("images", {"_id": img_id})
            else:
                img_doc = await db.mongo_find_one("images", {"_id": img_id})
            
            if img_doc:
                if img_doc.get("base64Data"):
                    return img_doc["base64Data"]
                elif img_doc.get("file_path"):
                    file_path = img_doc["file_path"]
                    if os.path.exists(file_path):
                        async with aiofiles.open(file_path, "rb") as f:
                            image_bytes = await f.read()
                        base64_encoded = base64_module.b64encode(image_bytes).decode('utf-8')
                        content_type = img_doc.get("content_type", "image/jpeg")
                        return f"data:{content_type};base64,{base64_encoded}"
        except Exception as e:
            logger.error(f"Failed to load option image {img_id}: {e}")
        return None
    
    def normalize_b64(b64: str) -> str:
        if b64 and not b64.startswith("data:image"):
            return f"data:image/png;base64,{b64}"
        return b64
    
    # Check enhancedOptions for image options
    enh = q.get("enhancedOptions") or []
    for i, opt in enumerate(enh):
        label = chr(65 + i)  # A, B, C, D...
        if isinstance(opt, dict) and opt.get("type") == "image":
            b64 = opt.get("base64Data")
            if not b64:
                b64 = await load_image_by_id(opt.get("id"))
            if b64:
                option_images.append({
                    "option": label,
                    "image": normalize_b64(b64)
                })
    
    # Also check regular options array for image objects
    opts = q.get("options") or []
    for i, opt in enumerate(opts):
        label = chr(65 + i)
        if isinstance(opt, dict):
            if opt.get("type") == "image" or opt.get("image"):
                b64 = opt.get("base64Data") or opt.get("image")
                if not b64:
                    b64 = await load_image_by_id(opt.get("id"))
                if b64:
                    option_images.append({
                        "option": label,
                        "image": normalize_b64(b64)
                    })
    
    if option_images:
        logger.info(f"📷 Extracted {len(option_images)} option images: {[o['option'] for o in option_images]}")
    
    return option_images

def _normalize_choice_text(s: str) -> str:
    import re as _re
    t = (s or '').upper().strip()
    # Support any single letter A-Z for MCQ options (not just A-D)
    m = _re.search(r"\b([A-Z])\b", t)
    return m.group(1) if m else t

def _normalize_numeric_text(s: str) -> str:
    t = (s or '').strip().replace(' ', '').replace(',', '.')
    if ':' in t and t.count(':') == 1 and all(part.isdigit() for part in t.split(':')):
        t = t.replace(':', '.')
    return t

def require_student_or_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require student, admin, or B2C user access"""
    allowed_types = ["student", "admin", "b2c_user", "b2c_admin"]
    if current_user.get("user_type") not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student, admin, or B2C user access required"
        )
    return current_user

@router.post("/next")
@limiter.limit("60/minute")
async def get_next_practice_question(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """Return a random next question from MongoDB.

    Behavior:
    - Prefer questions tagged as Practice Sets in metadata when available
    - Fall back gracefully to all questions if tag is missing
    - Build figure/image payloads with base64 when available; else serve via images API
    """
    try:
        import random
        from pydantic import BaseModel
        
        class NextQuestionRequest(BaseModel):
            subject: Optional[str] = None
            difficulty: Optional[str] = None
            excludeIds: Optional[List[str]] = None

        # Safely parse request body (optional)
        subject: Optional[str] = None
        difficulty: Optional[str] = None
        exclude_ids: List[str] = []
        try:
            if request.headers.get("content-type", "").startswith("application/json"):
                body = await request.json()
                if isinstance(body, dict):
                    req_data = NextQuestionRequest(**body)
                    subject = req_data.subject
                    difficulty = req_data.difficulty
                    if req_data.excludeIds:
                        exclude_ids = list(req_data.excludeIds)
        except Exception as _e:
            # Fall back to defaults if parsing fails; do not reject
            exclude_ids = []
        
        # Get questions from MongoDB (Practice Sets)
        fetched_ids = []
        metadatas = []

        admin_id = current_user.get("admin_id")
        admin_id_str = str(admin_id).strip() if admin_id is not None else ""

        def _apply_admin_scope(query_filter: Dict[str, Any]) -> None:
            """Attach admin isolation filter when available."""
            if not admin_id_str:
                return
            try:
                query_filter["admin_id"] = ObjectId(admin_id_str)
            except Exception:
                query_filter["admin_id"] = admin_id_str
        
        # Build filter for Practice Sets
        mongo_filter: Dict[str, Any] = {"document_type": "Practice Sets"}
        _apply_admin_scope(mongo_filter)
        if subject:
            mongo_filter["subject"] = subject
        if difficulty:
            mongo_filter["difficulty"] = difficulty

        mongo_questions = await db.mongo_find("questions", mongo_filter, limit=1000)
        fetched_ids = [q.get("id") for q in mongo_questions if q.get("id")]
        
        for q in mongo_questions:
            metadata = {
                "fullData": json.dumps(q, default=str),
                "subject": q.get("subject", ""),
                "difficulty": q.get("difficulty", "medium"),
                "document_type": q.get("document_type", "Practice Sets")
            }
            metadatas.append(metadata)
            
        logger.info(f"Fetched {len(fetched_ids)} Practice Sets questions from MongoDB (subject={subject}, difficulty={difficulty})")

        # Additional MongoDB fallback if no results
        if not fetched_ids:
            mongo_filter = {"metadata.document_type": "Practice Sets"}
            _apply_admin_scope(mongo_filter)
            if subject:
                mongo_filter["subject"] = subject
            if difficulty:
                mongo_filter["difficulty"] = difficulty

            mongo_questions = await db.mongo_find("questions", mongo_filter, projection={"id": 1})
            fetched_ids = [q.get("id") for q in mongo_questions if q.get("id")]
            logger.info(f"MongoDB fallback fetched {len(fetched_ids)} question ids")

        if not fetched_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No practice questions found. Please upload Practice Sets documents and process them."
            )

        # Refine the pool via fullData if available
        if metadatas and fetched_ids:
            refined: List[str] = []
            for qid, md in zip(fetched_ids, metadatas):
                full_json = md.get('fullData')
                if not full_json:
                    refined.append(qid)
                    continue
                try:
                    import json as _json
                    fd = _json.loads(full_json)
                    doc_type = (fd.get('metadata', {}) or {}).get('document_type')
                    if doc_type and doc_type != 'Practice Sets':
                        continue
                    if subject and fd.get('subject') != subject:
                        continue
                    if difficulty and fd.get('difficulty') != difficulty:
                        continue
                    refined.append(qid)
                except Exception:
                    refined.append(qid)
            fetched_ids = refined

        # Filter out excluded IDs
        available_ids = [qid for qid in fetched_ids if qid not in exclude_ids]
        
        if not available_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No new questions available. All questions have been attempted."
            )
        
        # Select random question from available
        question_id = random.choice(available_ids)
        
        # Get question from MongoDB
        question_doc = await db.mongo_find_one("questions", {"id": question_id}) or {}
        
        if not question_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question data not found"
            )
        
        # Build image URLs for frontend - images are served from local disk via /api/v1/images/{id}
        images_with_urls = []
        for img_ref in question_doc.get("images", []) or []:
            # Support both string ID and dict refs having id
            img_id = img_ref.get("id") if isinstance(img_ref, dict) else img_ref
            if not img_id:
                continue
            # Check if image exists in MongoDB using _id field (MongoDB primary key)
            img_doc = await db.mongo_find_one("images", {"_id": img_id})
            if img_doc:
                images_with_urls.append({
                    "id": img_id,
                    "url": f"/api/v1/images/{img_id}",  # Serve from local disk
                    "contentType": img_doc.get("content_type", "image/jpeg"),
                    "filename": img_doc.get("original_filename", str(img_id))
                })
            else:
                logger.warning(f"Image {img_id} referenced in question {question_id} but not found in database")
        
        # Also include QUESTION FIGURES (diagrams)
        figures_with_urls = []
        for fig_ref in question_doc.get("question_figures", []):
            try:
                fig_id = fig_ref.get("id") if isinstance(fig_ref, dict) else fig_ref
                base64_data = None

                # First check if base64Data is embedded in the figure reference
                if isinstance(fig_ref, dict) and fig_ref.get("base64Data"):
                    base64_data = fig_ref["base64Data"]
                    logger.info(f"Using embedded base64Data for figure {fig_id}")
                else:
                    # Try to get base64Data from images collection
                    img_doc = await db.mongo_find_one("images", {"_id": fig_id})
                    if img_doc:
                        # Check if base64Data is stored in the document
                        if img_doc.get("base64Data"):
                            base64_data = img_doc["base64Data"]
                            logger.info(f"Using stored base64Data for figure {fig_id}")
                        # If not, read from file_path and convert to base64
                        elif img_doc.get("file_path"):
                            import os
                            import base64
                            file_path = img_doc["file_path"]
                            if os.path.exists(file_path):
                                try:
                                    async with aiofiles.open(file_path, "rb") as f:
                                        image_bytes = await f.read()
                                    base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
                                    # Determine content type from file extension or stored content_type
                                    content_type = img_doc.get("content_type", "image/jpeg")
                                    if not content_type.startswith("image/"):
                                        # Default to jpeg if content type is not an image
                                        content_type = "image/jpeg"
                                    base64_data = f"data:{content_type};base64,{base64_encoded}"
                                    logger.info(f"✅ Loaded and converted image {fig_id} from file: {len(base64_data)} bytes")
                                except Exception as file_err:
                                    logger.error(f"❌ Failed to read image file {file_path}: {file_err}")
                            else:
                                logger.warning(f"⚠️ Image file not found: {file_path}")
                        else:
                            logger.warning(f"⚠️ No base64Data or file_path for image {fig_id}")
                    else:
                        logger.warning(f"⚠️ Image document not found: {fig_id}")

                figures_with_urls.append({
                    "id": fig_id,
                    "url": f"/api/v1/images/{fig_id}",
                    "contentType": "image/jpeg",
                    "filename": (fig_ref.get("filename") if isinstance(fig_ref, dict) else str(fig_id)),
                    "base64Data": base64_data,
                    "description": (fig_ref.get("description", "") if isinstance(fig_ref, dict) else ""),
                    "type": "diagram"
                })
            except Exception as _e:
                logger.error(f"❌ Practice figures processing error: {_e}", exc_info=True)

        merged_images = images_with_urls + figures_with_urls

        # Format LaTeX in question text and options
        from utils.latex_formatter import format_question_latex

        question = {
            "id": question_id,
            "text": question_doc.get("text", ""),
            "subject": question_doc.get("subject", ""),
            "difficulty": question_doc.get("difficulty", "medium"),
            "options": question_doc.get("options", []),
            "images": merged_images,  # Include both option images and figures
            "questionFigures": figures_with_urls,  # Separate field for diagrams/figures
            "enhancedOptions": question_doc.get("enhancedOptions"),
            "correctAnswer": question_doc.get("correctAnswer") or question_doc.get("correct_answer"),  # Include answer for debugging
            "metadata": question_doc.get("metadata", {})
        }

        # Format LaTeX expressions in question text and options
        question = format_question_latex(question)
        
        logger.info(f"Returning question {question_id}: {len(images_with_urls)} option images, {len(figures_with_urls)} figures")
        if figures_with_urls:
            for idx, fig in enumerate(figures_with_urls):
                logger.info(f"  Figure {idx + 1}: ID={fig.get('id')}, has_base64={bool(fig.get('base64Data'))}, base64_len={len(fig.get('base64Data', ''))}")
        
        return {
            "success": True,
            "question": question
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get next practice question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get next question: {str(e)}"
        )


class UploadedImageFile(BaseModel):
    """Model for uploaded image files from frontend"""
    data: str  # Base64 encoded image data
    name: str
    type: str  # MIME type


class UploadedDocumentFile(BaseModel):
    """Model for uploaded document files (PDF/DOCX) from frontend"""
    data: str  # Base64 encoded document data
    name: str
    type: str  # 'pdf' or 'docx'


class QuestionPageRefsModel(BaseModel):
    """Per-question page mapping from the Stoody Pen QuestionSession."""
    activePages: Optional[List[int]] = None        # Physical notebook page numbers
    bookType: Optional[str] = None                 # e.g. "LS", "MS"
    copyId: Optional[str] = None                   # Copy set ID
    timeIntervals: Optional[List[Dict[str, Any]]] = None  # [{startTs, endTs}]


class EvaluateRequest(BaseModel):
    questionId: str
    answerText: Optional[str] = None
    canvasData: Optional[str] = None
    canvasPages: Optional[List[str]] = None
    # Uploaded files from mobile/desktop file picker
    uploadedImages: Optional[List[UploadedImageFile]] = None  # Images uploaded directly
    uploadedDocuments: Optional[List[UploadedDocumentFile]] = None  # PDF/DOCX files
    # Optional tracking fields for practice history
    documentId: Optional[str] = None  # Practice set document ID
    sessionId: Optional[str] = None   # Practice session ID
    timeSpent: Optional[int] = None   # Time spent in seconds
    hintsUsed: Optional[int] = 0      # Number of hints used
    # Per-question page mapping from Stoody Pen (which pages + time intervals)
    questionPageRefs: Optional[QuestionPageRefsModel] = None

    # Be flexible: accept pages as strings or objects with common keys; normalize to data URLs
    @validator('canvasData', pre=True)
    def _normalize_canvas_data(cls, v):
        try:
            if v and isinstance(v, str) and not v.startswith('data:image'):
                return f"data:image/png;base64,{v}"
        except Exception:
            pass
        return v

    @validator('canvasPages', pre=True)
    def _normalize_canvas_pages(cls, v):
        if v is None:
            return v
        try:
            if isinstance(v, list):
                out: List[str] = []
                for item in v:
                    s = None
                    if isinstance(item, str):
                        s = item
                    elif isinstance(item, dict):
                        s = (
                            item.get('dataUrl')
                            or item.get('url')
                            or item.get('data')
                            or item.get('image')
                            or item.get('src')
                        )
                    if s:
                        if not s.startswith('data:image'):
                            s = f"data:image/png;base64,{s}"
                        out.append(s)
                return out
            # If a single string is provided, wrap as list
            if isinstance(v, str):
                s = v
                if not s.startswith('data:image'):
                    s = f"data:image/png;base64,{s}"
                return [s]
        except Exception:
            return v
        return v

    @root_validator(pre=True)
    def _coerce_aliases(cls, values):
        # Accept snake_case aliases from frontend
        mapping = {
            'question_id': 'questionId',
            'answer_text': 'answerText',
            'canvas_data': 'canvasData',
            'canvas_pages': 'canvasPages',
            'document_id': 'documentId',
            'session_id': 'sessionId',
            'time_spent': 'timeSpent',
            'hints_used': 'hintsUsed',
            'question_page_refs': 'questionPageRefs',
        }
        for src, dst in mapping.items():
            if src in values and dst not in values:
                values[dst] = values[src]
        # If canvasPages provided as single object/string elsewhere, normalize to list
        cp = values.get('canvasPages')
        if isinstance(cp, str):
            values['canvasPages'] = [cp]
        return values

class EvaluateResponse(BaseModel):
    success: bool = True
    evaluation: Dict[str, Any]


@router.post("/evaluate", response_model=EvaluateResponse)
@limiter.limit("120/minute")
async def evaluate_submission(
    request: Request,
    payload: EvaluateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """Evaluate student's submission (canvas image and/or text) for a question with AI tutor feedback.
    
    ENHANCED VERSION: Uses multi-stage OCR pipeline for reliable handwriting recognition:
    1. Image enhancement (upscaling, contrast, stroke thickening)
    2. Dedicated OCR extraction with confidence scoring
    3. Fallback strategies for low-confidence results
    4. Improved prompting for handwriting analysis

    Returns: { success, evaluation: { correct, score, extractedAnswer, feedback, reasoning, ocrConfidence } }
    """
    try:
        import json as _json
        import re as _re
        import ast as _ast
        
        qid = payload.questionId
        answer_text = (payload.answerText or "").strip()
        canvas_data = payload.canvasData
        
        # Detect if user is B2C (uses B2C database)
        user_type = current_user.get("user_type", "")
        is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"
        
        logger.info(f"📝 Evaluating submission for Q:{qid}, user_type:{user_type}, is_b2c:{is_b2c}")
        
        # Normalize canvas data header if client sent raw base64
        if canvas_data and not canvas_data.startswith("data:image"):
            canvas_data = f"data:image/png;base64,{canvas_data}"

        # Fetch question from MongoDB
        # For B2C users, use B2C database
        question_doc = await _load_question_doc(db, qid, is_b2c=is_b2c)
        if not question_doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")

        # Pull correct answer
        ca_primary = question_doc.get("correctAnswer")
        ca_alt = question_doc.get("correct_answer")
        correct_answer = str((ca_primary if ca_primary is not None else (ca_alt if ca_alt is not None else ""))).strip()

        # Resolve option letter to actual value (e.g. "A" -> "7 days")
        resolved = _resolve_correct_answer(correct_answer, question_doc)
        correct_answer_value = resolved["resolved_value"]   # The actual answer content
        correct_answer_display = resolved["display"]         # Human-readable: "A (7 days)"
        is_option_letter = resolved["is_option_letter"]      # True if answer is A/B/C/D

        # Extract question text and options
        question_text = str(question_doc.get("text", ""))
        options_text = _options_text_from_question(question_doc)
        
        # Check stored question_type (set during upload or admin edit)
        stored_question_type = (question_doc.get("question_type") or "").lower().strip()

        # Determine if this is MCQ: must have options AND not be explicitly marked subjective
        is_mcq = bool(options_text) and stored_question_type != "subjective"

        # For subjective/non-MCQ questions: if the stored correctAnswer is a single option
        # letter (A-J) but there are no options to resolve it against, it's a meaningless
        # OCR artifact. Clear it so the LLM solves the question itself.
        if not is_mcq and is_option_letter and correct_answer_value == correct_answer:
            logger.info(
                f"Clearing bogus option-letter correctAnswer='{correct_answer}' for non-MCQ Q:{qid} "
                f"(no options to resolve against — LLM will solve instead)."
            )
            correct_answer = ""
            correct_answer_value = ""
            correct_answer_display = ""
            is_option_letter = False

        # Initialize AI service.
        # SWM-011: All LLM calls (text + vision) are routed through the shared
        # gate (pcr_practice).  The direct OpenAI service is only used as a
        # fallback when the gate module is not deployed.
        from services.async_openai_service import AsyncOpenAIService
        ai = AsyncOpenAIService()  # fallback only when gate is unavailable

        # Prepare images: Question Figures + Option Images + Student Canvas
        # 1. Question Figures (diagrams, inline images)
        question_images = await _figure_images_base64(question_doc, db, is_b2c)
        
        # 2. Option Images (for MCQs where options are images)
        option_images_data = await _option_images_base64(question_doc, db, is_b2c)
        option_images = [oi["image"] for oi in option_images_data]
        
        # Combine all question-related images
        all_question_images = question_images + option_images
        
        logger.info(f"📷 Total question images: {len(all_question_images)} (figures: {len(question_images)}, option images: {len(option_images)})")
        
        # 2. Student Canvas Images + Uploaded Images - ENHANCED PROCESSING
        student_images_raw = []
        if payload.canvasPages and len(payload.canvasPages) > 0:
            student_images_raw = payload.canvasPages
        elif canvas_data:
            student_images_raw = [canvas_data]
        
        # Add uploaded images to student submission
        if payload.uploadedImages:
            for uploaded_img in payload.uploadedImages:
                try:
                    img_data = uploaded_img.data
                    logger.info(f"📎 Processing uploaded image: {uploaded_img.name}, type: {uploaded_img.type}, data starts with: {img_data[:50]}...")
                    # Ensure proper data URL format
                    if not img_data.startswith('data:'):
                        # If it doesn't start with data:, add the proper prefix
                        img_data = f"data:{uploaded_img.type};base64,{img_data.split(',')[-1] if ',' in img_data else img_data}"
                    student_images_raw.append(img_data)
                    logger.info(f"✅ Added uploaded image: {uploaded_img.name}")
                except Exception as img_err:
                    logger.error(f"❌ Failed to process uploaded image {uploaded_img.name}: {img_err}")
            logger.info(f"📎 Total uploaded images added: {len(payload.uploadedImages)}, student_images_raw count: {len(student_images_raw)}")
        
        # Process uploaded documents (PDF/DOCX) - extract text
        uploaded_doc_text = ""
        if payload.uploadedDocuments:
            for doc in payload.uploadedDocuments:
                try:
                    doc_text = await _extract_text_from_document(doc.data, doc.type, doc.name)
                    if doc_text:
                        uploaded_doc_text += f"\n[From {doc.name}]:\n{doc_text}\n"
                        logger.info(f"📄 Extracted {len(doc_text)} chars from {doc.name}")
                except Exception as doc_err:
                    logger.warning(f"Failed to extract text from {doc.name}: {doc_err}")
        
        # === STAGE 1: IMAGE ENHANCEMENT ===
        # Enhance canvas images for better LLM vision analysis.
        # No intermediate OCR model is used — the raw (enhanced) images go directly
        # to the unbiased extraction (Stage 2A) and final evaluation (Stage 2B)
        # so that GPT-5.1's own vision reads the handwriting without being biased
        # by a potentially wrong OCR transcript.
        
        if student_images_raw:
            logger.info(f"🖼️ Processing {len(student_images_raw)} student images...")
            try:
                from utils.image_processor import enhance_canvas_images_batch, is_canvas_empty

                # Filter out blank/empty canvas pages before enhancement to save tokens
                non_empty_images = []
                for idx, img_url in enumerate(student_images_raw):
                    if img_url and not is_canvas_empty(img_url):
                        non_empty_images.append(img_url)
                    else:
                        logger.info(f"📷 Skipping blank canvas page {idx + 1}")

                if not non_empty_images:
                    logger.warning("⚠️ All canvas pages are blank after filtering")
                    # Keep at least the raw images so evaluation can report "no answer"
                    non_empty_images = [img for img in student_images_raw if img]
                else:
                    logger.info(f"📷 {len(non_empty_images)}/{len(student_images_raw)} pages have content")

                logger.info(f"🖼️ Enhancing {len(non_empty_images)} student images...")
                try:
                    enhanced_student_images = enhance_canvas_images_batch(non_empty_images, target_width=1500)
                    if not enhanced_student_images:
                        logger.warning("⚠️ Image enhancement returned empty list, using raw images")
                        enhanced_student_images = non_empty_images
                except Exception as enhance_err:
                    logger.warning(f"⚠️ Image enhancement failed: {enhance_err}. Using raw images.")
                    enhanced_student_images = non_empty_images

                # Use enhanced images for subsequent LLM evaluation
                student_images = enhanced_student_images

            except ImportError as ie:
                logger.warning(f"Image processor not available: {ie}. Using raw images.")
                student_images = student_images_raw
            except Exception as img_err:
                logger.error(f"Image processing failed: {img_err}. Continuing with raw images.")
                student_images = student_images_raw
        else:
            student_images = []
            logger.info("📷 No student images to process")
        
        # === STAGE 2: COMBINED EVALUATION WITH ENHANCED PROMPT ===
        # Combine typed answer with uploaded document text (no intermediate OCR injection)
        combined_answer = answer_text
        
        # Add uploaded document text to combined answer
        if uploaded_doc_text:
            if combined_answer:
                combined_answer = f"{combined_answer}\n\n[Uploaded Document Content]:{uploaded_doc_text}"
            else:
                combined_answer = f"[Uploaded Document Content]:{uploaded_doc_text}"
            logger.info(f"📄 Added uploaded document text to combined answer")
        
        # Combine all images for the LLM - STUDENT IMAGES FIRST, then question images
        # This ensures the LLM focuses on the student's work first
        all_images = student_images + all_question_images
        num_q_images = len(all_question_images)
        num_fig_images = len(question_images)
        num_opt_images = len(option_images)
        num_s_images = len(student_images)
        
        # Log question details for debugging - DETAILED IMAGE LOGGING
        logger.info(f"📝 Question {qid}: text_len={len(question_text)}, correct='{correct_answer}', is_mcq={is_mcq}")
        logger.info(f"📷 Image breakdown for Q:{qid}:")
        logger.info(f"   - Student canvas images: {num_s_images} (lengths: {[len(img) if img else 0 for img in student_images[:3]]}...)")
        logger.info(f"   - Question figures: {num_fig_images} (from question_figures + images arrays)")
        logger.info(f"   - Option images: {num_opt_images} (from enhancedOptions with type=image)")
        if question_images:
            logger.info(f"   - Question image samples: {[img[:50]+'...' if img else 'None' for img in question_images[:2]]}")
        if not all_question_images and not student_images:
            logger.warning(f"⚠️ No images at all for Q:{qid} evaluation! This may cause issues.")
        
        # Determine if we need the LLM to solve the question itself
        has_correct_answer = bool(correct_answer and correct_answer.strip())
        
        # === LANGUAGE DETECTION FOR MULTILINGUAL SUPPORT ===
        # Detect question language to ensure LLM responds in the same language
        detected_language = detect_language(question_text)
        language_instruction = get_language_instruction(detected_language)
        logger.info(f"🌐 Language detection: question='{detected_language}' for Q:{qid}")

        # Detect if this is an essay/descriptive question (common in commerce, humanities)
        _essay_keywords = [
            'distinguish', 'explain', 'describe', 'discuss', 'compare', 'contrast',
            'define', 'elaborate', 'enumerate', 'critically examine', 'analyze', 'analyse',
            'what are', 'what is', 'how does', 'why is', 'state', 'mention',
            'differentiate', 'illustrate', 'comment', 'evaluate', 'justify',
        ]
        # Long-form math/science questions also need higher token budgets
        _longform_math_keywords = [
            'find', 'solve', 'integrate', 'differentiate', 'prove', 'derive',
            'calculate', 'compute', 'show that', 'verify', 'simplify',
            'factorise', 'factorize', 'sketch', 'draw', 'construct',
        ]
        # Math symbols that indicate a computational problem (questions may
        # contain only the symbol without a keyword like "integrate")
        _math_symbols = ['∫', '∑', '∂', '∏', '∮', 'lim ', 'd/dx', 'd/dt']
        _q_lower = question_text.lower()
        is_essay_question = (
            not is_mcq and
            any(kw in _q_lower for kw in _essay_keywords)
        )
        _has_math_symbol = any(sym in question_text for sym in _math_symbols)
        is_longform_question = is_essay_question or (
            not is_mcq and
            (any(kw in _q_lower for kw in _longform_math_keywords) or _has_math_symbol)
        )
        # If the question is purely mathematical (detected via symbols), it is
        # NOT an essay even if it happens to contain a keyword like "evaluate".
        if _has_math_symbol and is_essay_question:
            is_essay_question = False
            logger.info(f"📐 Math symbol detected — overriding essay classification for Q:{qid}")

        # === STAGE 2A: UNBIASED EXTRACTION (no correct answer shown) ===
        unbiased_extracted_answer = ""
        unbiased_transcribed_text = ""
        unbiased_extraction_confidence = 0.0
        unbiased_extraction_source = "none"

        if answer_text:
            unbiased_extracted_answer = answer_text
            unbiased_extraction_confidence = 1.0
            unbiased_extraction_source = "typed_answer"

        if num_s_images > 0:
            extraction_prompt = (
                "Extract ONLY what the student wrote from the handwritten submission.\n"
                "Do NOT solve the question and do NOT infer the likely correct option.\n"
                "Return strict JSON:\n"
                "{\n"
                '  "final_answer": "student final answer if visible, else empty",\n'
                '  "transcribed_text": "readable transcription of student content",\n'
                '  "confidence": 0.0,\n'
                '  "notes": "short extraction notes"\n'
                "}\n\n"
                "Question context (for symbol interpretation only):\n"
                f"{_truncate_for_prompt(question_text, 1200)}\n\n"
            )
            if options_text:
                extraction_prompt += (
                    "Options context (for letter mapping only; do not infer):\n"
                    f"{_truncate_for_prompt(options_text, 1200)}\n\n"
                )
            extraction_prompt += "Analyze only what is visibly present in student handwriting."

            extraction_system_prompt = (
                "You are a strict OCR extractor. Never hallucinate and never guess based on expected correctness."
            )

            try:
                extraction_max = 1200 if is_essay_question else 500
                extraction_resp = await _gate_vision_call(
                    db, current_user, student_images,
                    extraction_prompt,
                    system_prompt=extraction_system_prompt,
                    max_tokens=extraction_max,
                )
                extraction_raw = (extraction_resp.get("response") or "").strip()
                extraction_parsed = robust_json_parse(extraction_raw) or {}
                if isinstance(extraction_parsed, dict):
                    extracted_final = str(
                        extraction_parsed.get("final_answer")
                        or extraction_parsed.get("extracted_answer")
                        or extraction_parsed.get("selected_option")
                        or extraction_parsed.get("detected_text")
                        or ""
                    ).strip()
                    transcribed = str(
                        extraction_parsed.get("transcribed_text")
                        or extraction_parsed.get("work_shown")
                        or extraction_parsed.get("notes")
                        or ""
                    ).strip()
                    parsed_conf = _coerce_float(
                        extraction_parsed.get("confidence", extraction_parsed.get("ocr_confidence")),
                        default=0.0
                    )

                    if transcribed:
                        unbiased_transcribed_text = transcribed
                    if extracted_final and not unbiased_extracted_answer:
                        unbiased_extracted_answer = extracted_final
                        unbiased_extraction_source = "vision_extraction"
                        unbiased_extraction_confidence = max(unbiased_extraction_confidence, parsed_conf)
                    elif not unbiased_extracted_answer and transcribed and is_essay_question:
                        unbiased_extracted_answer = transcribed
                        unbiased_extraction_source = "vision_transcription"
                        unbiased_extraction_confidence = max(unbiased_extraction_confidence, parsed_conf)
            except Exception as extraction_err:
                logger.warning(f"⚠️ Unbiased extraction failed: {extraction_err}")


        if not unbiased_extracted_answer and combined_answer:
            unbiased_extracted_answer = _truncate_for_prompt(combined_answer, 2000)
            unbiased_extraction_source = "combined_text"
            unbiased_extraction_confidence = max(unbiased_extraction_confidence, 0.3)

        # Backward-compatible OCR fields are derived from the unbiased extraction stage.
        # This keeps downstream response contracts stable without reintroducing a biased OCR pass.
        ocr_extracted_text = (unbiased_transcribed_text or unbiased_extracted_answer or "").strip()
        ocr_confidence = float(unbiased_extraction_confidence or 0.0)
        has_extracted_content = bool(
            (unbiased_extracted_answer and unbiased_extracted_answer.strip())
            or (unbiased_transcribed_text and unbiased_transcribed_text.strip())
        )

        uploaded_doc_excerpt = _truncate_for_prompt(uploaded_doc_text, 12000)
        
        # Construct IMPROVED Prompt with clearer structure and explicit instructions
        prompt = (
            "You are an expert tutor evaluating a student's handwritten answer. "
            "Your task is to determine if their answer is CORRECT or INCORRECT.\n\n"
        )
        
        # Add LANGUAGE INSTRUCTION at the very beginning of the prompt
        prompt += language_instruction
        
        # Add clear image labeling with detailed breakdown
        if num_s_images > 0 and num_q_images > 0:
            prompt += f"📷 IMAGE GUIDE: You will see {num_s_images + num_q_images} images total.\n"
            prompt += f"  - Images 1 to {num_s_images}: STUDENT'S HANDWRITTEN WORK (analyze these carefully)\n"
            img_offset = num_s_images + 1
            if num_fig_images > 0:
                prompt += f"  - Images {img_offset} to {img_offset + num_fig_images - 1}: QUESTION DIAGRAMS/FIGURES (essential for understanding the question)\n"
                img_offset += num_fig_images
            if num_opt_images > 0:
                prompt += f"  - Images {img_offset} to {img_offset + num_opt_images - 1}: MCQ OPTION IMAGES (these are the answer choices A, B, C, D as images)\n"
            prompt += "\n⚠️ IMPORTANT: You MUST examine the QUESTION DIAGRAMS to understand the problem correctly!\n\n"
        elif num_s_images > 0:
            prompt += f"📷 IMAGE GUIDE: You will see {num_s_images} image(s) of the STUDENT'S HANDWRITTEN WORK.\n\n"
        elif num_q_images > 0:
            prompt += f"📷 IMAGE GUIDE: You will see {num_q_images} image(s) of QUESTION DIAGRAMS.\n\n"
        
        # Question section
        prompt += "═══════════════════════════════════════\n"
        prompt += "📚 QUESTION:\n"
        prompt += "═══════════════════════════════════════\n"
        prompt += f"{question_text}\n\n"
        
        # Options if MCQ
        if options_text:
            prompt += "📋 OPTIONS:\n"
            prompt += f"{options_text}\n\n"
        
        # Student's input section
        prompt += "═══════════════════════════════════════\n"
        prompt += "✍️ STUDENT'S SUBMISSION:\n"
        prompt += "═══════════════════════════════════════\n"
        
        if answer_text:
            prompt += f"Typed Answer: {answer_text}\n"
        if unbiased_extracted_answer:
            prompt += f"Unbiased Extracted Answer: {unbiased_extracted_answer}\n"
        if unbiased_transcribed_text:
            prompt += f"Unbiased Transcription: {_truncate_for_prompt(unbiased_transcribed_text, 3000)}\n"

        if uploaded_doc_excerpt:
            prompt += f"Uploaded Document Content:\n{uploaded_doc_excerpt}\n"
        if num_s_images > 0:
            prompt += f"Handwritten Canvas: {num_s_images} page(s) submitted - EXAMINE CAREFULLY.\n"
        if not answer_text and not has_extracted_content and not uploaded_doc_excerpt and num_s_images == 0:
            prompt += "(No answer submitted)\n"
        prompt += "\n"

        # Correct answer section after student extraction context to reduce anchoring bias.
        prompt += "═══════════════════════════════════════\n"
        prompt += "✅ REFERENCE ANSWER FOR EVALUATION:\n"
        prompt += "═══════════════════════════════════════\n"
        if has_correct_answer:
            if is_mcq and is_option_letter:
                # For MCQ: show BOTH the letter AND the resolved option content
                prompt += f"Correct Option Letter: {correct_answer}\n"
                if correct_answer_value != correct_answer:
                    prompt += f"Option {correct_answer} Content/Value: {correct_answer_value}\n"
                prompt += "\n"
            else:
                prompt += f"{correct_answer}\n\n"
        else:
            prompt += "⚠️ NO CORRECT ANSWER PROVIDED BY ADMIN\n\n"
            prompt += "🧠 YOU MUST SOLVE THIS QUESTION YOURSELF:\n"
            if is_mcq:
                prompt += "1. Read the question carefully and analyze each option.\n"
                prompt += "2. Determine which option (A, B, C, or D) is correct.\n"
                prompt += "3. Use this as your reference to evaluate the student's answer.\n\n"
            else:
                prompt += "1. Read the question carefully.\n"
                prompt += "2. Solve it step-by-step to find the correct answer.\n"
                prompt += "3. Use your solution as the reference to evaluate the student's answer.\n\n"
            logger.warning(f"⚠️ Question {qid} has no correct_answer stored! LLM must solve it.")

        # Specific evaluation instructions based on question type AND whether answer is provided
        if is_mcq:
            prompt += "🎯 EVALUATION RULES (Multiple Choice Question):\n"
            prompt += "1. Look for a LETTER (A, B, C, D) or the VALUE/CONTENT of the correct option in the student's answer.\n"
            prompt += "2. IMPORTANT: Students may write EITHER the option letter OR the actual answer value — BOTH ARE CORRECT.\n"
            if has_correct_answer and is_option_letter and correct_answer_value != correct_answer:
                prompt += f"   - Example: Writing '{correct_answer}' OR '{correct_answer_value}' are BOTH CORRECT answers.\n"
            prompt += "3. Students often show their WORK (calculations, equations, diagrams) before writing their final answer.\n"
            prompt += "4. If you see calculations/work but NO final letter/answer, evaluate if their work leads to the correct answer.\n"
            if has_correct_answer:
                prompt += f"5. The student is CORRECT if:\n"
                prompt += f"   - Their final letter matches '{correct_answer}', OR\n"
                if correct_answer_value != correct_answer:
                    prompt += f"   - Their final answer matches the value '{correct_answer_value}', OR\n"
                prompt += f"   - Their calculations/work correctly lead to the correct option\n"
            else:
                prompt += "5. Compare the student's letter/work to YOUR solved answer.\n"
            prompt += "6. For numerical MCQs: If the student writes the correct numerical value, mark CORRECT even without a letter.\n\n"
        elif is_essay_question:
            prompt += "🎯 EVALUATION RULES (Essay/Descriptive Question):\n"
            prompt += "1. Transcribe ALL text from the student's handwriting carefully.\n"
            prompt += "2. This is an ESSAY/DESCRIPTIVE question - evaluate the CONTENT and KEY POINTS.\n"
            prompt += "3. KEY EVALUATION CRITERIA for essay questions:\n"
            prompt += "   - Did the student address the main question?\n"
            prompt += "   - Did they include relevant points/definitions/concepts?\n"
            prompt += "   - Is their explanation coherent and logical?\n"
            prompt += "   - For 'distinguish/compare' questions: Did they mention differences/similarities?\n"
            if has_correct_answer:
                prompt += f"4. Compare their answer to: '{correct_answer[:200]}{'...' if len(correct_answer) > 200 else ''}'.\n"
            else:
                prompt += "4. You MUST provide the IDEAL ANSWER in 'solved_answer' field.\n"
                prompt += "   - For essay questions, write a brief model answer (key points only).\n"
            prompt += "5. SCORING for essays:\n"
            prompt += "   - 1.0 (correct): Covers most key points adequately\n"
            prompt += "   - 0.7-0.9: Good answer with minor gaps\n"
            prompt += "   - 0.4-0.6: Partial answer, some key points missing\n"
            prompt += "   - 0.1-0.3: Attempted but significantly incomplete\n"
            prompt += "   - 0.0: No relevant content or completely wrong\n\n"
        else:
            prompt += "🎯 EVALUATION RULES (Subjective Question):\n"
            prompt += "1. Transcribe all text, numbers, and equations from the student's handwriting.\n"
            prompt += "2. Look for their FINAL answer (often boxed, circled, or underlined).\n"
            if has_correct_answer:
                prompt += f"3. Compare their answer to the correct answer: '{correct_answer_value[:100]}{'...' if len(correct_answer_value) > 100 else ''}'.\n"
            else:
                prompt += "3. You MUST SOLVE this question yourself first, then compare to the student's answer.\n"
            prompt += "4. ⚠️ SEMANTIC EQUIVALENCE RULES (CRITICAL — FOLLOW STRICTLY):\n"
            prompt += "   - '9' and 'nine' and '9.0' and 'Nine' are ALL THE SAME → mark CORRECT\n"
            prompt += "   - '7' and 'seven' and '7.00' are ALL THE SAME → mark CORRECT\n"
            prompt += "   - '1/2' and '0.5' and 'half' are ALL THE SAME → mark CORRECT\n"
            prompt += "   - '100' and 'hundred' and 'one hundred' are THE SAME → mark CORRECT\n"
            prompt += "   - Units can differ: '7 days' and '7' are the same if the question asks for days\n"
            prompt += "   - Case does NOT matter: 'Delhi' = 'delhi' = 'DELHI'\n"
            prompt += "   - Minor spelling variations are acceptable if the meaning is clear\n"
            prompt += "   - Compare the MEANING and VALUE, NOT the exact string format\n"
            prompt += "5. They are CORRECT if the answer is mathematically or semantically equivalent.\n"
            prompt += "6. Partial credit (score 0.5): if they're on the right track but made a small error.\n\n"
        
        
        # JSON output instructions - simplified and clearer
        prompt += "═══════════════════════════════════════\n"
        prompt += "📊 OUTPUT FORMAT - RETURN VALID JSON ONLY:\n"
        prompt += "═══════════════════════════════════════\n"
        prompt += "```json\n"
        prompt += "{\n"

        # Use different examples based on question type
        if is_mcq:
            prompt += '  "extracted_answer": "B",\n'
            prompt += '  "work_shown": "Student calculated using formula...",\n'
            prompt += '  "is_correct": false,\n'
            prompt += '  "score": 0.0,\n'
            if not has_correct_answer:
                prompt += '  "solved_answer": "C",\n'
        elif is_essay_question:
            prompt += '  "extracted_answer": "Financial Accounting focuses on external users, Management Accounting focuses on internal users...",\n'
            prompt += '  "work_shown": "Student mentioned key differences including...",\n'
            prompt += '  "is_correct": true,\n'
            prompt += '  "score": 0.8,\n'
            if not has_correct_answer:
                prompt += '  "solved_answer": "Key points: 1) Financial Accounting - external reporting, GAAP compliance. 2) Management Accounting - internal decision-making, no mandatory standards.",\n'
        else:
            prompt += '  "extracted_answer": "42",\n'
            prompt += '  "work_shown": "Brief summary of student work",\n'
            prompt += '  "is_correct": false,\n'
            prompt += '  "score": 0.0,\n'
            if not has_correct_answer:
                prompt += '  "solved_answer": "45",\n'

        prompt += '  "what_went_wrong": "Explanation if wrong",\n'
        prompt += '  "correct_solution": "Step by step solution",\n'
        prompt += '  "feedback": "Encouraging feedback",\n'
        prompt += '  "reasoning": "Your evaluation logic"\n'
        prompt += "}\n"
        prompt += "```\n\n"

        prompt += "📝 FIELD GUIDELINES:\n"
        if is_mcq:
            prompt += "- extracted_answer: Just the LETTER (A, B, C, or D).\n"
        elif is_essay_question:
            prompt += "- extracted_answer: A BRIEF summary of what the student wrote (key points only).\n"
        else:
            prompt += "- extracted_answer: For numerical, the number. For text, a brief summary.\n"
        prompt += "- is_correct: Must be true or false (boolean, not string)\n"
        if is_essay_question:
            prompt += "- score: Use full range 0.0 to 1.0 for partial credit on essays\n"
        else:
            prompt += "- score: 0.0 (wrong), 0.5 (partial), 1.0 (correct)\n"
        if not has_correct_answer:
            if is_essay_question:
                prompt += "- solved_answer: REQUIRED - Write the key points of the ideal answer (2-4 sentences max)\n"
            else:
                prompt += "- solved_answer: The correct answer YOU determined (REQUIRED since no admin answer)\n"
        prompt += "\n"
        
        # Math formatting (simplified)
        prompt += "⚠️ MATH IN JSON STRINGS:\n"
        prompt += "Use LaTeX with properly escaped backslashes in JSON strings:\n"
        prompt += '- Write \\\\frac{a}{b} not \\frac{a}{b}\n'
        prompt += '- Write \\\\sqrt{x} not \\sqrt{x}\n'
        prompt += '- Write \\\\alpha, \\\\beta, \\\\tau (not α, β, τ Unicode)\n'
        prompt += "Example: \"The answer is \\\\\\\\(mv^2\\\\\\\\)\"\n\n"
        
        prompt += "⚠️ CRITICAL RULES:\n"
        prompt += "1. Output ONLY valid JSON - no text before or after\n"
        prompt += "2. Use double quotes for strings, not single quotes\n"
        prompt += "3. Boolean values are true/false (lowercase, no quotes)\n"
        prompt += "4. Escape special characters in strings: \\\" for quotes, \\\\ for backslash\n\n"
        
        logger.info(
            f"📤 Sending evaluation to LLM for Q:{qid}. Images: {len(all_images)} "
            f"({num_s_images} student + {num_q_images} question). "
            f"Extracted: '{unbiased_extracted_answer[:80] if unbiased_extracted_answer else ''}'. "
            f"Transcribed: '{ocr_extracted_text}'. Correct: '{correct_answer[:50] if correct_answer else 'NONE'}'"
        )
        
        # Call LLM with enhanced system prompt - varies based on whether we have a correct answer
        # Include language instruction in system prompt for stronger enforcement
        language_system_instruction = (
            "CRITICAL LANGUAGE RULE: If the question is in Hindi (Devanagari script), "
            "you MUST respond ENTIRELY in Hindi. If the question is in English, respond in English. "
            "Match the language of the question exactly. "
        ) if detected_language == 'hindi' else ""
        
        # LaTeX formatting instruction for both system prompts
        latex_instruction = (
            "MATH FORMATTING: For ALL math expressions, use LaTeX notation with \\( \\) delimiters. "
            "Example: \\(\\tau = I\\alpha\\), \\(F = ma\\), \\(\\frac{a}{b}\\). "
            "NEVER use plain Unicode symbols (τ, α, ×). ALWAYS use LaTeX (\\tau, \\alpha, \\times). "
        )
        
        if has_correct_answer:
            system_prompt = (
                f"{language_system_instruction}"
                f"{latex_instruction}"
                "You are an expert answer evaluator specializing in reading handwritten student work. "
                "CRITICAL: Determine if the student's answer is CORRECT or INCORRECT using all provided evidence. "
                "The student's answer extraction was generated in a separate unbiased stage; use it as a primary signal. "
                "Compare against the provided reference answer while still validating with the student's work. "
                "Always output ONLY valid JSON without markdown code blocks. "
                "Be generous in interpreting messy handwriting but strict in evaluating correctness."
            )
        else:
            # Different system prompt for essay vs numerical/MCQ questions
            if is_essay_question:
                system_prompt = (
                    f"{language_system_instruction}"
                    f"{latex_instruction}"
                    "You are an expert tutor evaluating ESSAY/DESCRIPTIVE answers. "
                    "CRITICAL: Since NO MODEL ANSWER was provided, you MUST: "
                    "1. First, determine the KEY POINTS that should be in a good answer. "
                    "2. Use extracted student evidence and handwritten work together. "
                    "3. Evaluate based on: content accuracy, completeness, and clarity. "
                    "4. Include 'solved_answer' with the KEY POINTS (2-4 bullet points). "
                    "SCORING GUIDELINES for essays: "
                    "- 1.0: Excellent - covers all key points accurately. "
                    "- 0.7-0.9: Good - covers most key points with minor gaps. "
                    "- 0.4-0.6: Partial - some correct points but missing important ones. "
                    "- 0.1-0.3: Weak - attempted but significantly incomplete/incorrect. "
                    "- 0.0: No relevant content. "
                    "Always output ONLY valid JSON without markdown code blocks. "
                    "Be fair in evaluating - partial credit for partial answers."
                )
            else:
                system_prompt = (
                    f"{language_system_instruction}"
                    f"{latex_instruction}"
                    "You are an expert tutor who can both SOLVE questions AND evaluate student answers. "
                    "CRITICAL: Since NO CORRECT ANSWER was provided, you MUST first SOLVE the question yourself. "
                    "1. First, solve the question to determine the correct answer. "
                    "2. Then, interpret the student's extracted evidence and handwritten work together. "
                    "3. Compare the student's answer to YOUR solution. "
                    "4. Include your 'solved_answer' in the JSON response. "
                    "For MCQ, determine which letter (A/B/C/D) is correct, then verify if the student wrote that letter. "
                    "For subjective questions, solve it step-by-step, then compare to the student's work. "
                    "Always output ONLY valid JSON without markdown code blocks. "
                    "Be generous in interpreting messy handwriting but strict in evaluating correctness."
                )
        
        # Token budget: MCQ needs ~500 (letter match), non-MCQ long-form needs 2500+
        # because the LLM must write a full step-by-step solution in JSON.
        if is_mcq:
            eval_max_tokens = 800
        elif is_longform_question:
            eval_max_tokens = 2500
        else:
            eval_max_tokens = 1800

        # When no correct answer is stored, the LLM must solve the question itself.
        # Use temperature=0 for maximum determinism so repeated submissions get
        # consistent scores instead of random variation (0.2, 0.6, 0.7 for same work).
        eval_temperature = 0.0 if not has_correct_answer else 0.3

        if all_images:
            # SWM-011: Vision path — route through the shared gate (C4)
            response = await _gate_vision_call(
                db, current_user, all_images,
                prompt,
                system_prompt=system_prompt,
                max_tokens=eval_max_tokens,
                temperature=eval_temperature,
            )
        else:
            # SWM-011: Text-only path — route through the shared gate (C4)
            response = await _gate_text_call(
                db, current_user, prompt,
                system_prompt=system_prompt,
                max_tokens=eval_max_tokens,
                temperature=eval_temperature,
            )
            
        raw_response = (response.get("response") or "").strip()
        logger.info(f"📥 LLM Raw Response (first 500 chars): {raw_response[:500]}")
        
        # Parse JSON Response
        evaluation_data = {
            "correct": False,
            "score": 0.0,
            "extractedAnswer": unbiased_extracted_answer,
            "feedback": "",
            "reasoning": "",
            "answerSource": unbiased_extraction_source if unbiased_extracted_answer else "ai_eval",
            "extractionConfidence": unbiased_extraction_confidence,
            "ocrConfidence": ocr_confidence,
            "ocrExtractedText": ocr_extracted_text,
            "correctAnswer": correct_answer,  # Include for frontend display
            "correctAnswerSource": "admin_provided" if has_correct_answer else "unknown",
        }
        
        try:
            # Use the robust JSON parser that handles LaTeX, nested braces, and edge cases
            parsed = robust_json_parse(raw_response)
            
            if not parsed:
                logger.warning(f"⚠️ Initial JSON parse failed, attempting retry request...")

                # Customize retry prompt based on question type
                if is_essay_question:
                    retry_prompt = (
                        f"Based on the student's essay answer, provide ONLY this JSON:\n"
                        f'{{"is_correct": true, "score": 0.7, "extracted_answer": "brief summary of student answer", '
                        f'"solved_answer": "key points of ideal answer", "feedback": "evaluation feedback"}}'
                    )
                else:
                    retry_prompt = (
                        f"Based on the student's work shown earlier, provide ONLY this JSON (no explanation):\n"
                        f'{{"is_correct": true/false, "score": 0.0-1.0, "extracted_answer": "student answer", '
                        f'"feedback": "brief encouraging feedback", "reasoning": "your evaluation logic", '
                        f'"what_went_wrong": "explanation if wrong, else empty string", '
                        f'"correct_solution": "step by step solution"}}'
                    )

                retry_token_max = 600 if is_essay_question else 500

                if all_images:
                    # SWM-011: Vision retry — route through the shared gate (C4)
                    retry_response = await _gate_vision_call(
                        db, current_user, all_images,
                        retry_prompt,
                        system_prompt="You are a JSON generator. Output ONLY valid JSON, nothing else.",
                        max_tokens=retry_token_max,
                    )
                else:
                    # SWM-011: Text-only retry — route through gate (C4)
                    retry_response = await _gate_text_call(
                        db, current_user, retry_prompt,
                        system_prompt="You are a JSON generator. Output ONLY valid JSON.",
                        max_tokens=retry_token_max,
                    )

                retry_raw = (retry_response.get("response") or "").strip()
                logger.info(f"📥 Retry response: {retry_raw[:200]}")
                parsed = robust_json_parse(retry_raw)

                # If retry also fails, try to extract useful information from raw response
                if not parsed and raw_response:
                    logger.warning(f"⚠️ Retry also failed. Attempting text-based extraction from raw response...")
                    # For essay questions, attempt to provide partial evaluation
                    if is_essay_question:
                        # Extract any useful text from the raw LLM response
                        evaluation_data["feedback"] = raw_response[:1500]  # Use raw response as feedback
                        evaluation_data["reasoning"] = "Evaluation completed but JSON parsing failed. See feedback for details."
                        # For essay questions, give partial credit by default if student wrote something
                        if has_extracted_content or num_s_images > 0:
                            evaluation_data["score"] = 0.5  # Partial credit for attempted essay
                            evaluation_data["correct"] = False  # Can't verify without proper parsing
                        evaluation_data["answerSource"] = "text_extraction_fallback"
                        logger.info(f"📝 Essay fallback: Providing partial evaluation from raw response")
            
            if parsed and isinstance(parsed, dict):
                # Parse is_correct - handle various representations
                is_correct_val = parsed.get("is_correct", parsed.get("correct", False))
                if isinstance(is_correct_val, bool):
                    evaluation_data["correct"] = is_correct_val
                elif isinstance(is_correct_val, str):
                    evaluation_data["correct"] = is_correct_val.lower() in ("true", "yes", "correct", "1")
                else:
                    evaluation_data["correct"] = bool(is_correct_val)
                
                # Parse score - use LLM-provided score if available, else default based on correctness
                llm_score = parsed.get("score")
                if llm_score is not None:
                    try:
                        evaluation_data["score"] = float(llm_score)
                    except (ValueError, TypeError):
                        evaluation_data["score"] = 1.0 if evaluation_data["correct"] else 0.0
                else:
                    evaluation_data["score"] = 1.0 if evaluation_data["correct"] else 0.0
                
                parsed_extracted_answer = str(parsed.get("extracted_answer", "")).strip()
                if parsed_extracted_answer and not evaluation_data["extractedAnswer"]:
                    evaluation_data["extractedAnswer"] = parsed_extracted_answer
                    evaluation_data["answerSource"] = "llm_extraction"
                evaluation_data["feedback"] = str(parsed.get("feedback", "")).strip()
                evaluation_data["reasoning"] = str(parsed.get("reasoning", "")).strip()
                
                # Extract work_shown for display
                work_shown = parsed.get("work_shown", "")
                if work_shown:
                    evaluation_data["workShown"] = str(work_shown).strip()
                
                # Extract what_went_wrong - explanation of student's mistake
                what_went_wrong = parsed.get("what_went_wrong", "")
                if what_went_wrong:
                    evaluation_data["whatWentWrong"] = str(what_went_wrong).strip()
                
                # Extract correct_solution - step-by-step solution
                correct_solution = parsed.get("correct_solution", "")
                if correct_solution:
                    evaluation_data["correctSolution"] = str(correct_solution).strip()
                
                # Handle correct answer - either from admin or LLM
                solved_answer = parsed.get("solved_answer", "")
                if has_correct_answer:
                    # Admin provided answer — use human-readable display version
                    evaluation_data["correctAnswer"] = correct_answer_display  # e.g. "A (7)" instead of "A"
                    evaluation_data["correctAnswerSource"] = "admin_provided"
                    
                    # Validate: If LLM solved it differently, log a warning
                    if solved_answer:
                        sa_upper = solved_answer.strip().upper()
                        ca_upper = correct_answer.strip().upper()
                        # For MCQ check both letter and value
                        if is_option_letter:
                            if sa_upper != ca_upper and not _answers_are_equivalent(solved_answer.strip(), correct_answer_value):
                                logger.warning(f"⚠️ LLM's solved_answer '{solved_answer}' differs from admin's '{correct_answer}' (value='{correct_answer_value}')")
                        elif sa_upper != ca_upper:
                            logger.warning(f"⚠️ LLM's solved_answer '{solved_answer}' differs from admin's '{correct_answer}'")
                elif solved_answer:
                    # LLM solved the question.
                    # USER REQUEST: Do not display a short word/char "Correct Answer" box in the summary tab
                    # if the LLM solved it. Just let the detailed 'correctSolution' explain it.
                    evaluation_data["correctAnswer"] = ""
                    evaluation_data["correctAnswerSource"] = "llm_solved"
                    logger.info(f"🧠 LLM solved the question. Solved answer: '{solved_answer}' (hidden from short answer display)")
                else:
                    # No answer available - keep empty
                    evaluation_data["correctAnswer"] = ""
                    evaluation_data["correctAnswerSource"] = "unknown"
                    logger.warning(f"⚠️ No correct answer available for Q:{qid}")
                
                # If LLM didn't extract an answer but unbiased extraction did, use that result
                if not evaluation_data["extractedAnswer"] and ocr_extracted_text:
                    evaluation_data["extractedAnswer"] = ocr_extracted_text
                    evaluation_data["answerSource"] = "vision_extraction_fallback"
                
                # ──────────────────────────────────────────────
                # Robust post-LLM validation: catch contradictions
                # ──────────────────────────────────────────────
                extracted = evaluation_data["extractedAnswer"].strip() if evaluation_data["extractedAnswer"] else ""
                
                if is_mcq and extracted and correct_answer:
                    # MCQ validation: accept letter match OR value match
                    extracted_upper = extracted.upper()
                    ca_upper = correct_answer.strip().upper()
                    
                    # Check letter match (student wrote "A", correct is "A")
                    letter_match = (len(extracted_upper) == 1 and len(ca_upper) == 1 
                                    and extracted_upper == ca_upper)
                    # Check value match (student wrote "7", option A content is "7")
                    value_match = False
                    if correct_answer_value and correct_answer_value != correct_answer:
                        value_match = _answers_are_equivalent(extracted, correct_answer_value)
                    
                    is_match = letter_match or value_match
                    
                    if is_match != evaluation_data["correct"]:
                        logger.warning(
                            f"⚠️ MCQ contradiction detected! Extracted='{extracted}', "
                            f"Expected letter='{ca_upper}', value='{correct_answer_value}', "
                            f"letter_match={letter_match}, value_match={value_match}, "
                            f"LLM said correct={evaluation_data['correct']}. "
                            f"Overriding to match={is_match}"
                        )
                        evaluation_data["correct"] = is_match
                        evaluation_data["score"] = 1.0 if is_match else 0.0
                        
                elif not is_mcq and extracted and correct_answer_value:
                    # Subjective validation: semantic/numeric equivalence safety net
                    if _answers_are_equivalent(extracted, correct_answer_value):
                        if not evaluation_data["correct"]:
                            logger.warning(
                                f"⚠️ Subjective override: student='{extracted}' ≈ correct='{correct_answer_value}'. "
                                f"LLM incorrectly said wrong. Overriding to correct."
                            )
                            evaluation_data["correct"] = True
                            evaluation_data["score"] = 1.0
                
                logger.info(f"✅ JSON parsed successfully. is_correct={evaluation_data['correct']}, score={evaluation_data['score']}, extracted='{evaluation_data['extractedAnswer'][:50] if evaluation_data['extractedAnswer'] else 'EMPTY'}', correctAnswer='{evaluation_data['correctAnswer'][:50] if evaluation_data['correctAnswer'] else 'EMPTY'}', has_solution={bool(correct_solution)}")
                    
            else:
                # Fallback if no JSON found - provide meaningful evaluation anyway
                logger.warning(f"⚠️ Could not parse JSON from LLM response. Raw: {raw_response[:200]}")
                evaluation_data["feedback"] = raw_response[:2000] if raw_response else "Unable to evaluate. Please try again."
                evaluation_data["reasoning"] = "Evaluation completed but structured response parsing failed."

                # For essay questions, give partial credit if student attempted an answer
                if is_essay_question and (has_extracted_content or num_s_images > 0):
                    evaluation_data["score"] = 0.5  # Partial credit for attempted essay
                    evaluation_data["correct"] = False
                    evaluation_data["answerSource"] = "essay_fallback"
                    # Try to extract key information from raw response for essay feedback
                    if raw_response:
                        # Check if LLM mentioned correct/incorrect in its response
                        raw_lower = raw_response.lower()
                        if any(word in raw_lower for word in ['correct', 'right', 'good answer', 'well done', 'accurate']):
                            evaluation_data["score"] = 0.7
                        elif any(word in raw_lower for word in ['incorrect', 'wrong', 'missing', 'incomplete', 'needs improvement']):
                            evaluation_data["score"] = 0.3
                    logger.info(f"📝 Essay fallback: Assigned partial score {evaluation_data['score']} based on attempt")
                # Still use extraction output if available
                elif ocr_extracted_text:
                    evaluation_data["extractedAnswer"] = ocr_extracted_text
                    evaluation_data["answerSource"] = "vision_fallback"
                
        except Exception as parse_err:
            logger.error(f"❌ Failed to parse LLM evaluation JSON: {parse_err}")
            evaluation_data["feedback"] = raw_response
            evaluation_data["reasoning"] = f"JSON parse error: {parse_err}"
            if ocr_extracted_text:
                evaluation_data["extractedAnswer"] = ocr_extracted_text
                evaluation_data["answerSource"] = "vision_fallback"

        # ──────────────────────────────────────────────
        # Post-parse: ensure feedback & reasoning are never empty when we
        # have a valid score/correctness from the LLM.
        # ──────────────────────────────────────────────
        _score = evaluation_data.get("score", 0.0)
        _is_correct = evaluation_data.get("correct", False)
        _student_ans = evaluation_data.get("extractedAnswer", "")

        if not evaluation_data.get("feedback"):
            if _is_correct:
                evaluation_data["feedback"] = "Great work! Your answer is correct."
            elif _score >= 0.7:
                evaluation_data["feedback"] = "Good attempt! Your answer is mostly correct with minor issues."
            elif _score >= 0.4:
                evaluation_data["feedback"] = "You're on the right track, but there are some errors in your solution. Review the steps carefully."
            elif _student_ans:
                evaluation_data["feedback"] = "Your answer isn't quite right. Review the approach and try to identify where the mistake occurred."
            else:
                evaluation_data["feedback"] = "No answer was detected. Please write your solution on the canvas and submit again."
            logger.info(f"💬 Generated fallback feedback for Q:{qid} (score={_score})")

        if not evaluation_data.get("reasoning"):
            if _is_correct:
                evaluation_data["reasoning"] = "The student's answer matches the expected solution."
            elif _score > 0:
                evaluation_data["reasoning"] = f"The student's work was partially correct (score: {_score:.0%}). Some steps or the final answer contain errors."
            else:
                evaluation_data["reasoning"] = "The submitted answer does not match the expected solution."

        logger.info(f"🎯 Evaluation complete for Q:{qid}. Correct: {evaluation_data['correct']}, Extracted: '{evaluation_data['extractedAnswer']}', Expected: '{correct_answer[:30] if correct_answer else 'N/A'}', Source: {evaluation_data['answerSource']}")

        # === SAVE PRACTICE ATTEMPT TO DATABASE FOR HISTORY ===
        try:
            user_id = current_user.get("user_id") or current_user.get("student_id") or current_user.get("id")
            
            # Get document_id from question metadata or request payload
            doc_id = payload.documentId
            if not doc_id:
                # Try to extract from question metadata
                meta = question_doc.get("metadata", {}) or {}
                doc_id = meta.get("document_id") or meta.get("documentId") or question_doc.get("document_id")
            
            # Prepare attempt record
            q_type = "mcq" if is_mcq else ("essay" if is_essay_question else "numerical")

            # Build question_page_refs from the Stoody Pen QuestionSession
            question_page_refs = None
            if payload.questionPageRefs:
                qpr = payload.questionPageRefs
                if qpr.activePages:
                    question_page_refs = {
                        "active_pages": qpr.activePages,
                        "book_type": qpr.bookType,
                        "copy_id": qpr.copyId,
                        "time_intervals": qpr.timeIntervals or [],
                    }

            practice_attempt = {
                "student_id": str(user_id),
                "session_id": payload.sessionId,
                "document_id": doc_id,
                "question_id": qid,
                "question_text": question_text[:2000] if question_text else "",
                "question_type": q_type,
                "options": question_doc.get("options"),
                "student_answer": evaluation_data.get("extractedAnswer", ""),
                "correct_answer": correct_answer,
                "is_correct": evaluation_data.get("correct", False),
                "score": evaluation_data.get("score", 0.0),
                "time_spent": payload.timeSpent,
                "hints_used": payload.hintsUsed or 0,
                "evaluation_feedback": evaluation_data.get("feedback", "")[:2000] if evaluation_data.get("feedback") else "",
                "evaluation_reasoning": evaluation_data.get("reasoning", "")[:2000] if evaluation_data.get("reasoning") else "",
                "work_shown": evaluation_data.get("workShown", ""),
                "what_went_wrong": evaluation_data.get("whatWentWrong", ""),
                "correct_solution": evaluation_data.get("correctSolution", ""),
                "question_page_refs": question_page_refs,
                "created_at": datetime.utcnow(),
                "subject": question_doc.get("subject", ""),
                "difficulty": question_doc.get("difficulty", ""),
                "topic": (question_doc.get("metadata", {}) or {}).get("topic", ""),
            }
            
            # Save to appropriate database (B2C or main)
            if is_b2c:
                await db.b2c_insert_one("practice_attempts", practice_attempt)
            else:
                await db.mongo_insert_one("practice_attempts", practice_attempt)
            
            logger.info(f"💾 Saved practice attempt for student {user_id}, question {qid}")
            
        except Exception as save_err:
            # Log but don't fail the request if saving fails
            logger.error(f"❌ Failed to save practice attempt: {save_err}", exc_info=True)

        return EvaluateResponse(success=True, evaluation=evaluation_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to evaluate submission: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate submission"
        )

@router.post("/sessions", response_model=SessionResponse)
@limiter.limit("30/minute")
async def start_practice_session(
    request: Request,
    session_data: StartSessionRequest,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Start a new practice session"""
    try:
        user_id = current_user["user_id"]
        
        # Detect if user is B2C (uses B2C database)
        user_type = current_user.get("user_type", "")
        is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"

        # Create session record
        session_record = {
            "student_id": user_id,
            "mode": session_data.mode,
            "subject": session_data.subject,
            "difficulty": session_data.difficulty,
            "time_limit": session_data.time_limit,
            "document_id": session_data.document_id,  # Track which practice set
            "questions_attempted": 0,
            "correct_answers": 0,
            "total_time_spent": 0,
            "started_at": datetime.utcnow(),
            "is_completed": False,
            "questions": []  # Will store question attempts
        }

        # Use B2C database for B2C users
        if is_b2c:
            session_id = await db.b2c_insert_one("practice_sessions", session_record)
        else:
            session_id = await db.mongo_insert_one("practice_sessions", session_record)

        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start practice session"
            )

        return SessionResponse(
            id=session_id,
            mode=session_data.mode,
            subject=session_data.subject,
            difficulty=session_data.difficulty,
            questions_attempted=0,
            correct_answers=0,
            accuracy_rate=0.0,
            total_time_spent=0,
            started_at=session_record["started_at"],
            is_completed=False
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Start practice session error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start practice session"
        )

@router.post("/sessions/{session_id}/answer")
@limiter.limit("200/minute")
async def submit_session_answer(
    request: Request,
    session_id: str,
    answer_data: SessionAnswer,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Submit answer for a question in a practice session"""
    try:
        user_id = current_user["user_id"]

        # Get session
        session = await db.mongo_find_one("practice_sessions", {"_id": session_id})

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Practice session not found"
            )

        # Check ownership (students can only access their own sessions)
        if (current_user["user_type"] == "student" and
            session["student_id"] != user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        if session["is_completed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session is already completed"
            )

        # Get question to validate answer
        question = await db.mongo_find_one("questions", {"question_id": answer_data.question_id})

        # Validate answer
        is_correct = False
        score = 0
        if question:
            correct_answer = question.get("correct_answer", "")
            is_correct = (answer_data.answer.strip().lower() == correct_answer.strip().lower())
            if is_correct:
                score = question.get("points", 4.0)  # Default 4 marks per question

        # Create question attempt record in session
        question_attempt = {
            "question_id": answer_data.question_id,
            "answer": answer_data.answer,
            "is_correct": is_correct,
            "time_spent": answer_data.time_spent,
            "answered_at": datetime.utcnow()
        }

        # Update session with new attempt
        update_data = {
            "$push": {"questions": question_attempt},
            "$inc": {
                "questions_attempted": 1,
                "total_time_spent": answer_data.time_spent
            }
        }

        if is_correct:
            update_data["$inc"]["correct_answers"] = 1

        await db.mongo_update_one(
            "practice_sessions",
            {"_id": session_id},
            update_data
        )

        # Track in question_attempts collection for student monitoring
        if current_user["user_type"] == "student":
            try:
                student_oid = ObjectId(user_id)

                # Get admin_id from JWT token for data isolation
                admin_id = current_user.get("admin_id")
                if not admin_id:
                    logger.warning(f"Student {user_id} has no admin_id in JWT token")
                    admin_id = None

                # Insert into question_attempts collection
                attempt_doc = {
                    "student_id": student_oid,
                    "question_id": answer_data.question_id,
                    "session_id": session_id,
                    "answer": answer_data.answer,
                    "is_correct": is_correct,
                    "score": score,
                    "time_spent": answer_data.time_spent,
                    "created_at": datetime.utcnow(),
                    "metadata": {
                        "subject": question.get("subject") if question else None,
                        "difficulty": question.get("difficulty") if question else None
                    }
                }

                # Add admin_id for data isolation if available
                if admin_id:
                    attempt_doc["admin_id"] = admin_id

                await db.mongo_insert_one("question_attempts", attempt_doc)

                # Log activity in student_activity_log
                activity_doc = {
                    "student_id": student_oid,
                    "action": "question_attempted",
                    "timestamp": datetime.utcnow(),
                    "metadata": {
                        "question_id": answer_data.question_id,
                        "session_id": session_id,
                        "is_correct": is_correct,
                        "score": score,
                        "time_spent": answer_data.time_spent
                    }
                }

                # Add admin_id for data isolation if available
                if admin_id:
                    activity_doc["admin_id"] = admin_id

                await db.mongo_insert_one("student_activity_log", activity_doc)
            except Exception as e:
                logger.warning(f"Failed to track question attempt: {str(e)}")

        return {
            "message": "Answer submitted successfully",
            "is_correct": is_correct,
            "question_id": answer_data.question_id,
            "score": score
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Submit session answer error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit answer"
        )

@router.post("/sessions/{session_id}/complete")
@limiter.limit("30/minute")
async def complete_practice_session(
    request: Request,
    session_id: str,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Complete a practice session"""
    try:
        user_id = current_user["user_id"]

        # Get session
        session = await db.mongo_find_one("practice_sessions", {"_id": session_id})

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Practice session not found"
            )

        # Check ownership
        if (current_user["user_type"] == "student" and
            session["student_id"] != user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        if session["is_completed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session is already completed"
            )

        # Mark session as completed
        await db.mongo_update_one(
            "practice_sessions",
            {"_id": session_id},
            {
                "$set": {
                    "is_completed": True,
                    "completed_at": datetime.utcnow()
                }
            }
        )

        # Get updated session
        updated_session = await db.mongo_find_one("practice_sessions", {"_id": session_id})

        accuracy_rate = 0.0
        if updated_session["questions_attempted"] > 0:
            accuracy_rate = (updated_session["correct_answers"] / updated_session["questions_attempted"]) * 100

        return SessionResponse(
            id=session_id,
            mode=updated_session["mode"],
            subject=updated_session.get("subject"),
            difficulty=updated_session.get("difficulty"),
            questions_attempted=updated_session["questions_attempted"],
            correct_answers=updated_session["correct_answers"],
            accuracy_rate=round(accuracy_rate, 1),
            total_time_spent=updated_session["total_time_spent"],
            started_at=updated_session["started_at"],
            completed_at=updated_session["completed_at"],
            is_completed=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Complete practice session error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete practice session"
        )

@router.get("/sessions", response_model=SessionsListResponse)
@limiter.limit("60/minute")
async def get_practice_sessions(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    mode: Optional[str] = Query(None),
    is_completed: Optional[bool] = Query(None),
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Get practice sessions"""
    try:
        user_id = current_user["user_id"]
        user_type = current_user["user_type"]

        # Build filter
        filter_dict = {}
        if user_type == "student":
            filter_dict["student_id"] = user_id

        if mode:
            filter_dict["mode"] = mode
        if is_completed is not None:
            filter_dict["is_completed"] = is_completed

        # Get total count
        all_sessions = await db.mongo_find("practice_sessions", filter_dict)
        total_sessions = len(all_sessions)

        # Get paginated results
        skip = (page - 1) * limit
        sessions_data = await db.mongo_find(
            "practice_sessions",
            filter_dict,
            sort=[("started_at", -1)],
            skip=skip,
            limit=limit
        )

        sessions = []
        for session in sessions_data:
            accuracy_rate = 0.0
            if session["questions_attempted"] > 0:
                accuracy_rate = (session["correct_answers"] / session["questions_attempted"]) * 100

            sessions.append(SessionResponse(
                id=str(session["_id"]),
                mode=session["mode"],
                subject=session.get("subject"),
                difficulty=session.get("difficulty"),
                questions_attempted=session["questions_attempted"],
                correct_answers=session["correct_answers"],
                accuracy_rate=round(accuracy_rate, 1),
                total_time_spent=session["total_time_spent"],
                started_at=session["started_at"],
                completed_at=session.get("completed_at"),
                is_completed=session["is_completed"]
            ))

        return SessionsListResponse(
            sessions=sessions,
            total=total_sessions,
            page=page,
            limit=limit
        )

    except Exception as e:
        logger.error(f"Get practice sessions error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get practice sessions"
        )

@router.get("/stats", response_model=PracticeStats)
@limiter.limit("30/minute")
async def get_practice_stats(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Get practice statistics"""
    try:
        user_id = current_user["user_id"]
        user_type = current_user["user_type"]

        # Check cache first
        cache_key = f"practice_stats:{user_id}" if user_type == "student" else "practice_stats:admin"
        cached_stats = await cache.get(cache_key, "practice")
        if cached_stats:
            return PracticeStats(**cached_stats)

        # Build filter
        filter_dict = {}
        if user_type == "student":
            filter_dict["student_id"] = user_id

        # Get all sessions
        all_sessions = await db.mongo_find("practice_sessions", filter_dict)

        total_sessions = len(all_sessions)
        total_time_spent = sum(s.get("total_time_spent", 0) for s in all_sessions)

        # Calculate average accuracy
        completed_sessions = [s for s in all_sessions if s.get("is_completed", False)]
        total_accuracy = 0
        if completed_sessions:
            for session in completed_sessions:
                if session["questions_attempted"] > 0:
                    accuracy = (session["correct_answers"] / session["questions_attempted"]) * 100
                    total_accuracy += accuracy
            average_accuracy = total_accuracy / len(completed_sessions)
        else:
            average_accuracy = 0.0

        # Sessions by mode
        sessions_by_mode = {}
        for session in all_sessions:
            mode = session.get("mode", "unknown")
            sessions_by_mode[mode] = sessions_by_mode.get(mode, 0) + 1

        # Recent activity (last 7 days)
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        recent_sessions = [s for s in all_sessions if s["started_at"] >= recent_cutoff]
        recent_activity = [
            {
                "date": session["started_at"].date().isoformat(),
                "mode": session["mode"],
                "questions_attempted": session["questions_attempted"],
                "accuracy": round((session["correct_answers"] / session["questions_attempted"]) * 100, 1) if session["questions_attempted"] > 0 else 0
            }
            for session in recent_sessions[-10:]  # Last 10 recent sessions
        ]

        stats_data = {
            "total_sessions": total_sessions,
            "total_time_spent": total_time_spent,
            "average_accuracy": round(average_accuracy, 1),
            "sessions_by_mode": sessions_by_mode,
            "recent_activity": recent_activity
        }

        # Cache for 10 minutes
        await cache.set(cache_key, stats_data, 600, "practice")

        return PracticeStats(**stats_data)

    except Exception as e:
        logger.error(f"Practice stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get practice statistics"
        )


@router.post("/grade", response_model=EvaluateResponse)
@limiter.limit("120/minute")
async def grade_submission(
    request: Request,
    payload: EvaluateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """Backward-compatible alias for the canonical /evaluate pipeline."""
    return await evaluate_submission(
        request=request,
        payload=payload,
        current_user=current_user,
        db=db
    )

# Backward compatibility endpoint for older clients that used legacy evaluator wiring.
@router.post("/evaluate-compat", response_model=EvaluateResponse)
@limiter.limit("120/minute")
async def evaluate_submission_compat(
    request: Request,
    payload: EvaluateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    return await grade_submission(request, payload, current_user, db)


# =============================================================================
# PRACTICE ATTEMPT HISTORY ENDPOINTS
# =============================================================================

class PracticeAttemptResponse(BaseModel):
    """Single practice attempt record"""
    id: str
    student_id: str
    question_id: str
    question_text: Optional[str] = None
    question_type: Optional[str] = None
    options: Optional[List[str]] = None
    student_answer: Optional[str] = None
    correct_answer: Optional[str] = None
    is_correct: bool = False
    score: float = 0.0
    time_spent: Optional[int] = None
    hints_used: int = 0
    evaluation_feedback: Optional[str] = None
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    created_at: datetime
    document_id: Optional[str] = None

class PracticeHistoryStats(BaseModel):
    """Aggregated statistics for practice attempts"""
    total_attempted: int = 0
    total_correct: int = 0
    accuracy_percentage: float = 0.0
    avg_time_per_question: Optional[float] = None
    total_time_spent: int = 0

class PracticeHistoryResponse(BaseModel):
    """Response for practice attempt history"""
    success: bool = True
    attempts: List[Dict[str, Any]]
    total: int
    stats: PracticeHistoryStats

class PracticeSetStatsResponse(BaseModel):
    """Response for practice set statistics"""
    success: bool = True
    document_id: str
    question_stats: List[Dict[str, Any]]
    summary: Dict[str, Any]


@router.get("/attempts")
@limiter.limit("60/minute")
async def get_practice_attempts(
    request: Request,
    document_id: Optional[str] = Query(None, description="Filter by practice set document ID"),
    question_id: Optional[str] = Query(None, description="Filter by specific question ID"),
    subject: Optional[str] = Query(None, description="Filter by subject"),
    limit: int = Query(50, le=200, description="Maximum number of attempts to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get practice attempt history for the current student.
    
    Returns a list of past practice attempts with:
    - Question details
    - Student's answer
    - Correct/incorrect status
    - Evaluation feedback
    - Aggregated statistics
    """
    try:
        user_id = current_user.get("user_id") or current_user.get("student_id") or current_user.get("id")
        
        # Detect if user is B2C
        user_type = current_user.get("user_type", "")
        is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"
        
        # Build filter
        filter_dict = {"student_id": str(user_id)}
        if document_id:
            filter_dict["document_id"] = document_id
        if question_id:
            filter_dict["question_id"] = question_id
        if subject:
            filter_dict["subject"] = subject
        
        # Fetch attempts with pagination
        if is_b2c:
            attempts = await db.b2c_find(
                "practice_attempts",
                filter_dict,
                sort=[("created_at", -1)],
                limit=limit,
                skip=offset
            )
            total = await db.b2c_count("practice_attempts", filter_dict)
        else:
            attempts = await db.mongo_find(
                "practice_attempts",
                filter_dict,
                sort=[("created_at", -1)],
                limit=limit,
                skip=offset
            )
            total = await db.mongo_count("practice_attempts", filter_dict)
        
        # Convert ObjectId to string for JSON serialization
        attempt_list = []
        for a in attempts:
            attempt_dict = dict(a)
            if "_id" in attempt_dict:
                attempt_dict["id"] = str(attempt_dict.pop("_id"))
            if "created_at" in attempt_dict and attempt_dict["created_at"]:
                attempt_dict["created_at"] = attempt_dict["created_at"].isoformat()
            attempt_list.append(attempt_dict)
        
        # Calculate stats
        total_correct = sum(1 for a in attempt_list if a.get("is_correct"))
        total_time = sum(a.get("time_spent", 0) or 0 for a in attempt_list)
        
        stats = PracticeHistoryStats(
            total_attempted=total,
            total_correct=total_correct,
            accuracy_percentage=(total_correct / total * 100) if total > 0 else 0.0,
            avg_time_per_question=(total_time / len(attempt_list)) if attempt_list else None,
            total_time_spent=total_time
        )
        
        return {
            "success": True,
            "attempts": attempt_list,
            "total": total,
            "stats": stats.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get practice attempts error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch practice attempts"
        )


@router.get("/attempts/{attempt_id}")
@limiter.limit("60/minute")
async def get_practice_attempt_detail(
    request: Request,
    attempt_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get detailed information about a specific practice attempt.
    
    Returns the full attempt record including:
    - Original question with all details
    - Student's answer
    - Correct answer
    - Full AI feedback and reasoning
    - Work shown, what went wrong, correct solution
    """
    try:
        user_id = current_user.get("user_id") or current_user.get("student_id") or current_user.get("id")
        
        # Detect if user is B2C
        user_type = current_user.get("user_type", "")
        is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"
        
        # Fetch the attempt
        from bson import ObjectId
        try:
            oid = ObjectId(attempt_id)
            filter_dict = {"_id": oid, "student_id": str(user_id)}
        except Exception:
            filter_dict = {"_id": attempt_id, "student_id": str(user_id)}
        
        if is_b2c:
            attempt = await db.b2c_find_one("practice_attempts", filter_dict)
        else:
            attempt = await db.mongo_find_one("practice_attempts", filter_dict)
        
        if not attempt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Practice attempt not found"
            )
        
        # Convert for JSON serialization
        attempt_dict = dict(attempt)
        if "_id" in attempt_dict:
            attempt_dict["id"] = str(attempt_dict.pop("_id"))
        if "created_at" in attempt_dict and attempt_dict["created_at"]:
            attempt_dict["created_at"] = attempt_dict["created_at"].isoformat()
        
        return {
            "success": True,
            "attempt": attempt_dict
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get practice attempt detail error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch practice attempt"
        )


@router.get("/documents/{document_id}/stats")
@limiter.limit("30/minute")
async def get_practice_set_stats(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get aggregated statistics for a practice set.
    
    Returns:
    - Per-question statistics (attempts, last correct, mastery status)
    - Summary (total questions, mastered, needs practice)
    """
    try:
        user_id = current_user.get("user_id") or current_user.get("student_id") or current_user.get("id")
        
        # Detect if user is B2C
        user_type = current_user.get("user_type", "")
        is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"
        
        # Aggregation pipeline
        pipeline = [
            {"$match": {
                "student_id": str(user_id),
                "document_id": document_id
            }},
            {"$sort": {"created_at": -1}},
            {"$group": {
                "_id": "$question_id",
                "attempts": {"$sum": 1},
                "correct_count": {"$sum": {"$cond": ["$is_correct", 1, 0]}},
                "last_attempt": {"$first": "$created_at"},
                "last_correct": {"$first": "$is_correct"},
                "last_answer": {"$first": "$student_answer"},
                "question_text": {"$first": "$question_text"},
                "difficulty": {"$first": "$difficulty"},
                "avg_time": {"$avg": {"$ifNull": ["$time_spent", 0]}}
            }}
        ]
        
        if is_b2c:
            stats = await db.b2c_aggregate("practice_attempts", pipeline)
        else:
            stats = await db.mongo_aggregate("practice_attempts", pipeline)
        
        stats_list = list(stats)
        
        # Convert for JSON serialization
        question_stats = []
        for s in stats_list:
            stat_dict = {
                "question_id": s.get("_id"),
                "attempts": s.get("attempts", 0),
                "correct_count": s.get("correct_count", 0),
                "last_attempt": s.get("last_attempt").isoformat() if s.get("last_attempt") else None,
                "last_correct": s.get("last_correct", False),
                "last_answer": s.get("last_answer"),
                "question_text": s.get("question_text", "")[:200] if s.get("question_text") else "",
                "difficulty": s.get("difficulty"),
                "avg_time": s.get("avg_time"),
                "mastered": s.get("last_correct", False) and s.get("correct_count", 0) >= 1
            }
            question_stats.append(stat_dict)
        
        # Summary statistics
        total_questions = len(question_stats)
        mastered = sum(1 for q in question_stats if q.get("mastered"))
        needs_practice = total_questions - mastered
        total_attempts = sum(q.get("attempts", 0) for q in question_stats)
        total_correct = sum(q.get("correct_count", 0) for q in question_stats)
        
        return {
            "success": True,
            "document_id": document_id,
            "question_stats": question_stats,
            "summary": {
                "total_questions": total_questions,
                "mastered": mastered,
                "needs_practice": needs_practice,
                "total_attempts": total_attempts,
                "total_correct": total_correct,
                "overall_accuracy": (total_correct / total_attempts * 100) if total_attempts > 0 else 0.0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get practice set stats error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch practice set statistics"
        )


@router.get("/teacher-feedback")
@limiter.limit("60/minute")
async def get_teacher_feedback_for_document(
    request: Request,
    document_id: str = Query(..., description="Practice set document ID"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """
    Get all teacher feedback for the current student's attempts on a document.
    Returns a map of question_id -> teacher_feedback so the student can view
    feedback from their teacher on each question.
    """
    try:
        user_id = current_user.get("user_id") or current_user.get("student_id") or current_user.get("id")

        # Detect if user is B2C
        user_type = current_user.get("user_type", "")
        is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"

        filter_dict = {
            "student_id": str(user_id),
            "document_id": document_id,
            "teacher_feedback": {"$exists": True, "$ne": None},
        }

        if is_b2c:
            attempts = await db.b2c_find(
                "practice_attempts",
                filter_dict,
                sort=[("created_at", -1)],
            )
        else:
            attempts = await db.mongo_find(
                "practice_attempts",
                filter_dict,
                sort=[("created_at", -1)],
            )

        # Build a map keyed by question_id (keep only the latest per question)
        feedback_map: dict = {}
        for a in attempts:
            qid = a.get("question_id", "")
            if qid and qid not in feedback_map:
                tf = a.get("teacher_feedback", {})
                created_at = tf.get("created_at")
                updated_at = tf.get("updated_at")
                feedback_map[qid] = {
                    "question_id": qid,
                    "question_text": (a.get("question_text", "")[:200]
                                      if a.get("question_text") else ""),
                    "teacher_feedback": {
                        "text": tf.get("text", ""),
                        "tutor_name": tf.get("tutor_name", ""),
                        "created_at": created_at.isoformat()
                        if hasattr(created_at, "isoformat") else str(created_at or ""),
                        "updated_at": updated_at.isoformat()
                        if hasattr(updated_at, "isoformat") else str(updated_at or ""),
                    },
                }

        return {
            "success": True,
            "document_id": document_id,
            "feedback": list(feedback_map.values()),
            "total": len(feedback_map),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get teacher feedback error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch teacher feedback",
        )
