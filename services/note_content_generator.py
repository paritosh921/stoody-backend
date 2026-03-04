"""
Note Content Generator for Stoody

Generates flashcards and practice questions from classified note pages
using stored OCR text (no re-OCR needed). Supports delta regeneration.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import HTTPException
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CONTENT_GEN_MODEL = os.getenv("CONTENT_GEN_MODEL", "gpt-4o-mini")

MIN_PAGES_FOR_REGENERATION = 5


# ---------------------------------------------------------------------------
# Flashcard generation
# ---------------------------------------------------------------------------

async def generate_flashcards(
    user_id: str,
    subject: str,
    topic: str,
    texts: List[str],
    pages: List[Dict[str, Any]],
    tenant_db,
) -> Dict[str, Any]:
    """Generate flashcards from OCR text and store in note_flashcards."""
    combined_text = "\n---\n".join(t[:1500] for t in texts if t.strip())[:8000]

    prompt = f"""Generate educational flashcards from these handwritten {subject} notes on "{topic}".

Notes content:
{combined_text}

Generate 8-15 flashcards. Each flashcard should have:
- "front": A clear question or concept prompt
- "back": A concise, accurate answer
- "difficulty": "easy", "medium", or "hard"

Respond with JSON only:
{{"flashcards": [{{"front": "...", "back": "...", "difficulty": "..."}}]}}"""

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    resp = await client.chat.completions.create(
        model=CONTENT_GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        temperature=0.3,
    )

    raw = resp.choices[0].message.content or ""
    cards = _parse_json_response(raw, "flashcards")

    # Add IDs
    for card in cards:
        card["id"] = str(uuid.uuid4())[:8]

    source_page_keys = [
        {"pen_mac": p.get("pen_mac"), "book_type": p.get("book_type"), "page_number": p.get("page_number")}
        for p in pages
    ]

    now = datetime.utcnow()
    doc = {
        "flashcard_set_id": str(uuid.uuid4()),
        "user_id": user_id,
        "subject": subject,
        "topic": topic,
        "flashcards": cards,
        "source_page_keys": source_page_keys,
        "source_page_count": len(pages),
        "generated_at": now,
        "regenerated_at": None,
        "model_used": CONTENT_GEN_MODEL,
        "tokens_used": resp.usage.total_tokens if resp.usage else 0,
    }

    await tenant_db["note_flashcards"].update_one(
        {"user_id": user_id, "subject": subject, "topic": topic},
        {"$set": doc},
        upsert=True,
    )

    return doc


# ---------------------------------------------------------------------------
# Practice question generation
# ---------------------------------------------------------------------------

async def generate_practice(
    user_id: str,
    subject: str,
    topic: str,
    texts: List[str],
    pages: List[Dict[str, Any]],
    tenant_db,
) -> Dict[str, Any]:
    """Generate practice questions and store in note_practice_sets + documents."""
    combined_text = "\n---\n".join(t[:1500] for t in texts if t.strip())[:8000]

    prompt = f"""Generate practice questions from these handwritten {subject} notes on "{topic}".

Notes content:
{combined_text}

Generate 5-8 questions, mix of objective (MCQ) and subjective:
- Objective: 4 options, one correct answer, brief explanation
- Subjective: question text, model answer, key points

Respond with JSON only:
{{"questions": [
  {{"type": "objective", "text": "...", "options": ["A...", "B...", "C...", "D..."], "correct_answer": "A...", "explanation": "...", "difficulty": "medium"}},
  {{"type": "subjective", "text": "...", "model_answer": "...", "explanation": "...", "difficulty": "medium"}}
]}}"""

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    resp = await client.chat.completions.create(
        model=CONTENT_GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.3,
    )

    raw = resp.choices[0].message.content or ""
    questions = _parse_json_response(raw, "questions")

    for q in questions:
        q["id"] = str(uuid.uuid4())[:8]

    source_page_keys = [
        {"pen_mac": p.get("pen_mac"), "book_type": p.get("book_type"), "page_number": p.get("page_number")}
        for p in pages
    ]

    now = datetime.utcnow()
    practice_set_id = str(uuid.uuid4())

    doc = {
        "practice_set_id": practice_set_id,
        "user_id": user_id,
        "subject": subject,
        "topic": topic,
        "title": f"{topic} - Practice",
        "questions": questions,
        "source_page_keys": source_page_keys,
        "source_page_count": len(pages),
        "generated_at": now,
        "regenerated_at": None,
        "model_used": CONTENT_GEN_MODEL,
        "tokens_used": resp.usage.total_tokens if resp.usage else 0,
    }

    await tenant_db["note_practice_sets"].update_one(
        {"user_id": user_id, "subject": subject, "topic": topic},
        {"$set": doc},
        upsert=True,
    )

    # Also insert into documents collection for Practice tab integration
    await tenant_db["documents"].update_one(
        {"document_id": practice_set_id},
        {"$set": {
            "document_id": practice_set_id,
            "document_type": "Practice Sets",
            "source": "student_notes",
            "student_id": user_id,
            "title": f"{topic} - Practice",
            "subject": subject,
            "extracted_questions_count": len(questions),
            "created_at": now,
            "updated_at": now,
        }},
        upsert=True,
    )

    return doc


# ---------------------------------------------------------------------------
# Delta regeneration
# ---------------------------------------------------------------------------

async def regenerate_content(
    user_id: str,
    subject: str,
    topic: str,
    content_type: str,
    tenant_db,
) -> Dict[str, Any]:
    """Delta regeneration: append content from new pages only."""

    # Get current pages for topic
    current_pages = await tenant_db["note_classifications"].find(
        {"user_id": user_id, "subject": subject, "topic": topic},
        {"pen_mac": 1, "book_type": 1, "page_number": 1, "ocr_text": 1},
    ).to_list(100)

    current_keys = {
        (p["pen_mac"], p.get("book_type", "A5"), p["page_number"])
        for p in current_pages
    }

    # Get existing generated content
    collection = "note_flashcards" if content_type == "flashcards" else "note_practice_sets"
    existing = await tenant_db[collection].find_one(
        {"user_id": user_id, "subject": subject, "topic": topic}
    )

    if not existing:
        raise HTTPException(
            status_code=400,
            detail=f"No existing {content_type} to regenerate. Generate first.",
        )

    source_keys = {
        (k["pen_mac"], k.get("book_type", "A5"), k["page_number"])
        for k in existing.get("source_page_keys", [])
    }

    new_keys = current_keys - source_keys
    if len(new_keys) < MIN_PAGES_FOR_REGENERATION:
        raise HTTPException(
            status_code=400,
            detail=f"Need {MIN_PAGES_FOR_REGENERATION}+ new pages, only {len(new_keys)} found",
        )

    # Get OCR text for new pages only
    new_texts = [
        p.get("ocr_text", "")
        for p in current_pages
        if (p["pen_mac"], p.get("book_type", "A5"), p["page_number"]) in new_keys
        and p.get("ocr_text")
    ]

    if not new_texts:
        raise HTTPException(status_code=400, detail="No OCR text available for new pages")

    # Generate additional content
    new_content = await _generate_additional(new_texts, subject, topic, content_type)

    # Append to existing
    now = datetime.utcnow()
    if content_type == "flashcards":
        for card in new_content:
            card["id"] = str(uuid.uuid4())[:8]
        existing_items = existing.get("flashcards", [])
        existing_items.extend(new_content)
        update_field = "flashcards"
        update_items = existing_items
    else:
        for q in new_content:
            q["id"] = str(uuid.uuid4())[:8]
        existing_items = existing.get("questions", [])
        existing_items.extend(new_content)
        update_field = "questions"
        update_items = existing_items

    all_source_keys = [
        {"pen_mac": k[0], "book_type": k[1], "page_number": k[2]}
        for k in current_keys
    ]

    await tenant_db[collection].update_one(
        {"user_id": user_id, "subject": subject, "topic": topic},
        {"$set": {
            update_field: update_items,
            "source_page_keys": all_source_keys,
            "source_page_count": len(current_keys),
            "regenerated_at": now,
        }},
    )

    return {
        "new_items_added": len(new_content),
        "total_items": len(update_items),
        "new_pages_processed": len(new_keys),
    }


async def _generate_additional(
    texts: List[str],
    subject: str,
    topic: str,
    content_type: str,
) -> List[Dict[str, Any]]:
    """Generate additional flashcards or questions from new page texts."""
    combined = "\n---\n".join(t[:1500] for t in texts if t.strip())[:5000]

    if content_type == "flashcards":
        prompt = f"""Generate additional flashcards from these NEW {subject} notes on "{topic}".
Notes: {combined}

Generate 5-8 NEW flashcards (don't repeat existing ones).
Respond JSON only: {{"flashcards": [{{"front": "...", "back": "...", "difficulty": "..."}}]}}"""
        key = "flashcards"
    else:
        prompt = f"""Generate additional practice questions from these NEW {subject} notes on "{topic}".
Notes: {combined}

Generate 3-5 NEW questions (mix of objective and subjective).
Respond JSON only: {{"questions": [
  {{"type": "objective", "text": "...", "options": ["A...", "B...", "C...", "D..."], "correct_answer": "A...", "explanation": "...", "difficulty": "medium"}},
  {{"type": "subjective", "text": "...", "model_answer": "...", "explanation": "...", "difficulty": "medium"}}
]}}"""
        key = "questions"

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    resp = await client.chat.completions.create(
        model=CONTENT_GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        temperature=0.3,
    )

    raw = resp.choices[0].message.content or ""
    return _parse_json_response(raw, key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_response(text: str, key: str) -> List[Dict[str, Any]]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        items = data.get(key, [])
        if isinstance(items, list):
            return items
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse {key} JSON: {e} — raw: {text[:300]}")

    return []
