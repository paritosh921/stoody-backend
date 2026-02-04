"""
Two-LLM Orchestrator for Question Generation.

Implements the Generator (LLM1) and Reviewer (LLM2) loop system
with retry logic, validation, and memory management.

Key features:
- One question at a time generation
- Strict JSON schema validation
- Automatic retry with reviewer feedback
- Memory/state management for context
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .rag_prompts import (
    GENERATOR_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    build_generator_prompt,
    build_reviewer_prompt,
    build_revision_prompt,
    format_rag_context,
    validate_question_draft,
    validate_review_result,
)
from .models.job import (
    QuestionDraft,
    ReviewResult,
    ReviewIssue,
    QuestionGenerationAttempt,
    GeneratedQuestionRecord,
    PaperGenerationState,
    IterationLog,
    QuestionGenerationDebugLog,
)
from .qdrant_service import get_qdrant_service
from .embedding_service import get_embedding_service
from .diagram_validation import validate_diagram_spec, build_diagram_revision_instructions
from services.kimi_service import get_kimi_service

logger = logging.getLogger(__name__)


class LLMOrchestratorError(Exception):
    """Base exception for LLM orchestrator errors."""
    pass


class LLMOrchestrator:
    """
    Orchestrates the two-LLM loop for question generation.
    
    LLM1 (Generator): Generates questions based on RAG context
    LLM2 (Reviewer): Reviews questions for quality and accuracy
    
    The loop continues until:
    - Question is approved by reviewer
    - Maximum iterations reached (question marked for manual review)
    """
    
    MAX_ITERATIONS = 3
    LLM_TEMPERATURE = 0.6
    LLM_MODEL = "kimi-k2.5"  # Kimi 2.5 as specified
    
    def __init__(self):
        """Initialize the orchestrator."""
        self._kimi_service = None
        self._qdrant_service = None
        self._embedding_service = None
        
    async def initialize(self) -> None:
        """Initialize required services."""
        self._kimi_service = get_kimi_service()
        self._qdrant_service = get_qdrant_service()
        self._embedding_service = get_embedding_service()
        
        await self._qdrant_service.initialize()
        await self._embedding_service.initialize()
        
        logger.info("LLM Orchestrator initialized")
    
    # =========================================================================
    # RAG Retrieval
    # =========================================================================
    
    async def retrieve_context(
        self,
        query: str,
        tenant_id: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 8,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Retrieve relevant context from Qdrant using RAG.
        
        Args:
            query: Search query (e.g., topic or concept)
            tenant_id: Tenant identifier for collection
            filters: Optional Qdrant filters (subject, class, pdf_id)
            top_k: Number of results to retrieve
            
        Returns:
            Tuple of (list of chunks, list of chunk IDs)
        """
        try:
            # Generate query embedding
            embedding_result = await self._embedding_service.embed_text(query)
            query_vector = embedding_result.embedding
            
            # Build filter conditions
            filter_conditions = None
            if filters:
                must_conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        must_conditions.append({
                            "key": key,
                            "match": {"any": value}
                        })
                    else:
                        must_conditions.append({
                            "key": key,
                            "match": {"value": value}
                        })
                if must_conditions:
                    filter_conditions = {"must": must_conditions}
            
            # Search Qdrant
            logger.debug(
                f"RAG search: tenant={tenant_id}, query='{query[:80]}...', "
                f"filters={filters}, top_k={top_k}"
            )

            results = await self._qdrant_service.search(
                tenant_id=tenant_id,
                query_vector=query_vector,
                top_k=top_k,
                score_threshold=0.3,  # Minimum relevance
                filter_conditions=filter_conditions,
            )

            chunk_ids = [r["id"] for r in results]

            logger.info(
                f"Retrieved {len(results)} chunks for query: {query[:50]}... "
                f"(filters={list(filters.keys()) if filters else 'none'})"
            )

            if not results and filters:
                # Enhanced debugging when filters return no results
                logger.warning(
                    f"No chunks found with filters {filters}. "
                    "Check if source_file_id values match what's stored in Qdrant."
                )
                
                # Check if the requested source_file_ids exist
                if "source_file_id" in filters:
                    requested_ids = filters["source_file_id"]
                    if isinstance(requested_ids, str):
                        requested_ids = [requested_ids]
                    
                    # Verify which IDs exist
                    existence_check = await self._qdrant_service.verify_source_files_exist(
                        tenant_id=tenant_id,
                        source_file_ids=requested_ids,
                    )
                    
                    missing_ids = [sfid for sfid, exists in existence_check.items() if not exists]
                    existing_ids = [sfid for sfid, exists in existence_check.items() if exists]
                    
                    if missing_ids:
                        logger.error(
                            f"SOURCE FILE IDs NOT FOUND IN QDRANT: {missing_ids}. "
                            f"These documents may not have been uploaded to the knowledge base. "
                            f"Use /api/v1/knowledge-base/upload to ingest documents first."
                        )
                    
                    if existing_ids:
                        logger.info(f"Source file IDs that DO exist: {existing_ids}")
                    
                    # Also log what source_file_ids ARE in the collection (for debugging)
                    available_ids = await self._qdrant_service.get_source_file_ids(
                        tenant_id=tenant_id,
                        limit=10,
                    )
                    if available_ids:
                        logger.info(
                            f"Available source_file_ids in collection (first 10): {available_ids}"
                        )
                    else:
                        logger.warning(
                            f"No documents found in Qdrant collection for tenant {tenant_id}. "
                            "The collection may be empty."
                        )

            return results, chunk_ids
            
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            raise LLMOrchestratorError(f"Failed to retrieve context: {e}")
    
    # =========================================================================
    # LLM Calls
    # =========================================================================
    
    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = None,
    ) -> str:
        """
        Make an LLM call with retry logic.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            temperature: LLM temperature (defaults to class setting)
            
        Returns:
            LLM response text
        """
        temp = temperature or self.LLM_TEMPERATURE
        max_retries = 3
        
        # Debug: Log prompt details
        logger.info(f"LLM call - system_prompt length: {len(system_prompt) if system_prompt else 0}")
        logger.info(f"LLM call - user_prompt length: {len(user_prompt) if user_prompt else 0}")
        if not user_prompt or not user_prompt.strip():
            logger.error(f"EMPTY USER PROMPT DETECTED! user_prompt repr: {repr(user_prompt)}")
        else:
            logger.debug(f"User prompt first 500 chars: {user_prompt[:500]}")
        
        for attempt in range(max_retries):
            try:
                # KimiService.chat_completion_async takes messages list + optional system_prompt
                messages = [{"role": "user", "content": user_prompt}]
                
                result = await self._kimi_service.chat_completion_async(
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=temp,
                    max_tokens=4000,
                )
                
                if not result.get("success"):
                    raise LLMOrchestratorError(f"Kimi API error: {result.get('error', 'Unknown error')}")
                
                return result.get("response", "")
                
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise LLMOrchestratorError(f"LLM call failed after {max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff  # Exponential backoff
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response with robust error handling.
        
        Handles various response formats including markdown code blocks,
        and attempts to fix common JSON malformation issues.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed JSON object
        """
        import re
        
        # Clean response
        text = response.strip()
        
        # Try to extract JSON from markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        
        # Remove any leading/trailing non-JSON text
        # Find the first { and last }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace:last_brace + 1]
        
        # First attempt: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed: {e}, attempting fixes...")
        
        # Attempt common fixes
        fixed_text = text
        
        # Fix 1: Remove trailing commas before closing braces/brackets
        fixed_text = re.sub(r',(\s*[}\]])', r'\1', fixed_text)
        
        # Fix 2: Escape unescaped control characters in strings
        # This handles newlines and tabs that may have leaked into string values
        def escape_control_chars(match):
            s = match.group(0)
            # Replace actual newlines/tabs with escaped versions
            s = s.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            return s
        
        # Match string values (rough approximation)
        fixed_text = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', escape_control_chars, fixed_text)
        
        # Fix 3: Remove any BOM or weird unicode chars at start
        fixed_text = fixed_text.lstrip('\ufeff\u200b')
        
        # Second attempt with fixes
        try:
            return json.loads(fixed_text)
        except json.JSONDecodeError as e2:
            logger.warning(f"Fixed JSON parse also failed: {e2}, trying more aggressive fixes...")
        
        # Fix 4: Try to handle truncated JSON by finding balanced braces
        brace_count = 0
        bracket_count = 0
        in_string = False
        escape_next = False
        valid_end = -1
        
        for i, char in enumerate(fixed_text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and bracket_count == 0:
                    valid_end = i
                    break
            elif char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
        
        if valid_end > 0:
            truncated_text = fixed_text[:valid_end + 1]
            try:
                return json.loads(truncated_text)
            except json.JSONDecodeError:
                pass
        
        # Fix 5: Try to close any unclosed braces/brackets
        if brace_count > 0 or bracket_count > 0:
            completion = ']' * bracket_count + '}' * brace_count
            try:
                return json.loads(fixed_text + completion)
            except json.JSONDecodeError:
                pass
        
        # All fixes failed
        logger.error(f"Failed to parse JSON after all fixes: {text[:500]}...")
        raise LLMOrchestratorError(f"Invalid JSON response from LLM. Original error at position mentioned may indicate truncated or malformed output.")
    
    # =========================================================================
    # Repetition Detection and Hard Correction
    # =========================================================================
    
    def _compute_content_hash(self, draft: QuestionDraft) -> str:
        """
        Compute a hash of the question content for repetition detection.
        
        Hashes question_text + diagram_spec to detect when LLM1 outputs
        the same content repeatedly despite feedback.
        """
        content = f"{draft.question_text}|{json.dumps(draft.diagram_spec, sort_keys=True) if draft.diagram_spec else ''}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _detect_repetition(
        self,
        current_hash: str,
        previous_hashes: List[str],
    ) -> Tuple[bool, int]:
        """
        Detect if the current output is a repetition of previous outputs.
        
        Returns:
            Tuple of (is_repetition, repetition_count)
        """
        repetition_count = previous_hashes.count(current_hash)
        is_repetition = current_hash in previous_hashes
        return is_repetition, repetition_count
    
    def _build_hard_correction_prompt(
        self,
        draft: QuestionDraft,
        review: ReviewResult,
        repetition_count: int,
        diagram_feedback: str,
    ) -> str:
        """
        Build a more forceful prompt when LLM1 keeps repeating the same output.
        
        This prompt explicitly states what must change and threatens rejection.
        """
        issues_text = "\n".join([
            f"- [{i.issue_type}]: {i.message}\n  REQUIRED FIX: {i.fix_instructions}"
            for i in review.issues
        ])
        
        return f"""**HARD CORRECTION MODE - YOU HAVE REPEATED THE SAME OUTPUT {repetition_count} TIMES**

Your previous outputs were REJECTED because they contained the SAME errors.
You MUST make DIFFERENT changes this time or this question will FAIL permanently.

=== WHAT IS WRONG ===
{issues_text}

{f"=== DIAGRAM ISSUES ==={chr(10)}{diagram_feedback}" if diagram_feedback else ""}

=== YOUR PREVIOUS OUTPUT (DO NOT REPEAT THIS) ===
Question: {draft.question_text[:300]}...
{f"Diagram type: {draft.diagram_spec.get('diagram_type') if draft.diagram_spec else 'None'}" if draft.diagram_required else ""}

=== MANDATORY CHANGES ===
You MUST:
1. REWRITE the question text to address clarity/accuracy issues
2. {"CHANGE the diagram_spec parameters to match the question" if draft.diagram_required else "N/A"}
3. Provide DIFFERENT values/examples than your previous attempts

If you output the same content again, this question will be marked as FAILED.

Output ONLY the corrected JSON - no explanations."""
    
    def _create_iteration_log(
        self,
        iteration: int,
        draft: QuestionDraft,
        review: Optional[ReviewResult],
        generator_prompt: str,
        generator_duration_ms: int,
        reviewer_duration_ms: int,
        is_revision: bool,
        is_hard_correction: bool,
        repetition_detected: bool,
        diagram_validation_errors: List[str],
    ) -> IterationLog:
        """Create a detailed iteration log for debugging."""
        return IterationLog(
            iteration_number=iteration,
            timestamp=datetime.utcnow(),
            generator_prompt_hash=hashlib.sha256(generator_prompt.encode()).hexdigest()[:16],
            generator_prompt_length=len(generator_prompt),
            generator_output_hash=self._compute_content_hash(draft),
            generator_output_length=len(draft.question_text),
            generator_output_preview=draft.question_text[:500],
            diagram_spec_json=json.dumps(draft.diagram_spec) if draft.diagram_spec else None,
            diagram_validation_errors=diagram_validation_errors,
            reviewer_prompt_length=0,  # Filled later if needed
            reviewer_approved=review.approved if review else False,
            reviewer_issues=[i.issue_type for i in review.issues] if review and review.issues else [],
            reviewer_feedback_summary="; ".join([i.message[:100] for i in review.issues[:3]]) if review and review.issues else "",
            is_revision=is_revision,
            is_hard_correction_mode=is_hard_correction,
            repetition_detected=repetition_detected,
            generator_duration_ms=generator_duration_ms,
            reviewer_duration_ms=reviewer_duration_ms,
        )
    
    # =========================================================================
    # Generator (LLM1)
    # =========================================================================
    
    async def generate_question(
        self,
        question_type: str,
        subject: str,
        grade: str,
        difficulty: str,
        marks: int,
        topic_focus: str,
        diagram_required: bool,
        rag_context: List[Dict[str, Any]],
        chunk_ids: List[str],
        memory_state: PaperGenerationState,
    ) -> QuestionDraft:
        """
        Generate a single question using LLM1.
        
        Args:
            question_type: Type of question (mcq, short, long, numerical)
            subject: Subject name
            grade: Grade level
            difficulty: Target difficulty
            marks: Marks for this question
            topic_focus: Topic to focus on
            diagram_required: Whether to include diagram
            rag_context: Retrieved chunks
            chunk_ids: Available chunk IDs
            memory_state: Current paper generation state
            
        Returns:
            Generated QuestionDraft
        """
        # Format context
        context_str = format_rag_context(rag_context)
        memory_context = memory_state.to_prompt_context()
        
        # Build prompt
        prompt = build_generator_prompt(
            question_type=question_type,
            subject=subject,
            grade=grade,
            difficulty=difficulty,
            marks=marks,
            topic_focus=topic_focus,
            diagram_required=diagram_required,
            rag_context=context_str,
            chunk_ids=chunk_ids,
            memory_context=memory_context,
        )
        
        # Call LLM
        response = await self._call_llm(
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        
        # Parse response
        draft_data = self._parse_json_response(response)
        
        # Validate
        is_valid, errors = validate_question_draft(draft_data)
        if not is_valid:
            logger.warning(f"Invalid question draft: {errors}")
            # Try to fix common issues
            if "marks" not in draft_data:
                draft_data["marks"] = marks
            if "difficulty" not in draft_data:
                draft_data["difficulty"] = difficulty
            if "question_type" not in draft_data:
                draft_data["question_type"] = question_type
            if "source_chunk_ids" not in draft_data:
                draft_data["source_chunk_ids"] = []
            if "diagram_required" not in draft_data:
                draft_data["diagram_required"] = diagram_required
        
        # Create QuestionDraft
        return QuestionDraft(
            question_text=draft_data.get("question_text", ""),
            question_type=draft_data.get("question_type", question_type),
            options=draft_data.get("options"),
            correct_answer=draft_data.get("correct_answer", ""),
            explanation=draft_data.get("explanation", ""),
            marks=draft_data.get("marks", marks),
            difficulty=draft_data.get("difficulty", difficulty),
            source_chunk_ids=draft_data.get("source_chunk_ids") or [],
            diagram_required=draft_data.get("diagram_required", diagram_required),
            diagram_tool=draft_data.get("diagram_tool"),
            diagram_type=draft_data.get("diagram_type"),
            diagram_spec=draft_data.get("diagram_spec"),
            diagram_rendering_notes=draft_data.get("diagram_rendering_notes"),
            validation_checks=draft_data.get("validation_checks") or [],
            topic=draft_data.get("topic"),
            bloom_level=draft_data.get("bloom_level"),
        )

    
    # =========================================================================
    # Reviewer (LLM2)
    # =========================================================================
    
    async def review_question(
        self,
        draft: QuestionDraft,
        rag_context: List[Dict[str, Any]],
        chunk_ids: List[str],
        question_type: str,
        target_difficulty: str,
        grade: str,
        is_fallback_mode: bool = False,
    ) -> ReviewResult:
        """
        Review a question using LLM2.
        
        Args:
            draft: Question draft to review
            rag_context: Original retrieved chunks
            chunk_ids: Available chunk IDs
            question_type: Expected question type
            target_difficulty: Target difficulty
            grade: Grade level
            is_fallback_mode: Whether using LLM knowledge fallback (no RAG context)
            
        Returns:
            ReviewResult with approval status and issues
        """
        # Format context
        context_str = format_rag_context(rag_context)
        
        # Convert draft to dict
        draft_dict = {
            "question_text": draft.question_text,
            "question_type": draft.question_type,
            "options": draft.options,
            "correct_answer": draft.correct_answer,
            "explanation": draft.explanation,
            "marks": draft.marks,
            "difficulty": draft.difficulty,
            "topic": draft.topic,
            "bloom_level": draft.bloom_level,
            "source_chunk_ids": draft.source_chunk_ids,
            "diagram_required": draft.diagram_required,
            "diagram_spec": draft.diagram_spec,
        }
        
        # Build prompt - pass fallback mode so reviewer knows not to check for hallucination
        prompt = build_reviewer_prompt(
            question_draft=draft_dict,
            rag_context=context_str,
            chunk_ids=chunk_ids,
            question_type=question_type,
            target_difficulty=target_difficulty,
            grade=grade,
            is_fallback_mode=is_fallback_mode,
        )
        
        # Call LLM
        response = await self._call_llm(
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        
        # Parse response
        result_data = self._parse_json_response(response)
        
        # Validate
        is_valid, errors = validate_review_result(result_data)
        if not is_valid:
            logger.warning(f"Invalid review result: {errors}")
            # Return a default rejection
            return ReviewResult(
                approved=False,
                issues=[ReviewIssue(
                    issue_type="format",
                    message="Reviewer response was invalid",
                    fix_instructions="Please regenerate the question",
                )],
            )
        
        # In fallback mode, filter out hallucination and source_chunk_ids issues
        issues_list = result_data.get("issues") or []
        if is_fallback_mode:
            filtered_issues = [
                issue for issue in issues_list
                if issue.get("type") not in ["hallucination", "source_chunk_ids"]
            ]
            # If all issues were hallucination/source_chunk_ids, approve the question
            if not filtered_issues and issues_list:
                logger.info("Fallback mode: Filtering out hallucination/source_chunk_ids issues, auto-approving")
                result_data["approved"] = True
            issues_list = filtered_issues
        
        # Create ReviewResult
        issues = [
            ReviewIssue(
                issue_type=issue.get("type", "unknown"),
                message=issue.get("message", ""),
                fix_instructions=issue.get("fix_instructions", ""),
            )
            for issue in issues_list
        ]
        
        return ReviewResult(
            approved=result_data.get("approved", False),
            issues=issues,
            required_changes=result_data.get("required_changes"),
            quality_score=result_data.get("quality_score"),
        )
    
    # =========================================================================
    # Revision (LLM1 with feedback)
    # =========================================================================
    
    async def revise_question(
        self,
        draft: QuestionDraft,
        review: ReviewResult,
        rag_context: List[Dict[str, Any]],
        question_type: str,
        marks: int,
        difficulty: str,
        diagram_required: bool,
        subject: str = "",
        diagram_feedback: str = "",
    ) -> QuestionDraft:
        """
        Revise a question based on reviewer feedback.
        
        Args:
            draft: Original question draft
            review: Reviewer feedback
            rag_context: Original retrieved chunks
            question_type: Question type
            marks: Marks for this question
            difficulty: Target difficulty
            diagram_required: Whether diagram is required
            subject: Subject for diagram validation
            diagram_feedback: Specific diagram correction feedback
            
        Returns:
            Revised QuestionDraft
        """
        # Format context
        context_str = format_rag_context(rag_context)
        
        # Convert draft to dict
        draft_dict = {
            "question_text": draft.question_text,
            "question_type": draft.question_type,
            "options": draft.options,
            "correct_answer": draft.correct_answer,
            "explanation": draft.explanation,
            "marks": draft.marks,
            "difficulty": draft.difficulty,
            "topic": draft.topic,
            "bloom_level": draft.bloom_level,
            "source_chunk_ids": draft.source_chunk_ids,
            "diagram_required": draft.diagram_required,
            "diagram_spec": draft.diagram_spec,
        }
        
        # Convert review to dict
        review_dict = {
            "approved": review.approved,
            "issues": [
                {
                    "type": issue.issue_type,
                    "message": issue.message,
                    "fix_instructions": issue.fix_instructions,
                }
                for issue in review.issues
            ],
            "required_changes": review.required_changes,
        }
        
        # Build prompt with diagram feedback
        prompt = build_revision_prompt(
            original_draft=draft_dict,
            review_result=review_dict,
            rag_context=context_str,
            question_type=question_type,
            marks=marks,
            difficulty=difficulty,
            diagram_required=diagram_required,
            subject=subject,
            diagram_feedback=diagram_feedback,
        )
        
        # Call LLM
        response = await self._call_llm(
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        
        # Parse response
        revised_data = self._parse_json_response(response)
        
        # Validate
        is_valid, errors = validate_question_draft(revised_data)
        if not is_valid:
            logger.warning(f"Invalid revised draft: {errors}")
        
        # Validate diagram spec if present
        if diagram_required and revised_data.get("diagram_spec"):
            diagram_validation = validate_diagram_spec(
                revised_data.get("diagram_spec"),
                subject,
                revised_data.get("question_text", ""),
            )
            if not diagram_validation.is_valid:
                logger.warning(f"Invalid diagram spec after revision: {diagram_validation.errors}")
                # Apply suggested corrections
                if diagram_validation.corrections:
                    spec = revised_data.get("diagram_spec", {})
                    params = spec.get("params", spec.get("parameters", {}))
                    for key, value in diagram_validation.corrections.items():
                        params[key] = value
                    spec["params"] = params
                    revised_data["diagram_spec"] = spec
        
        # Create QuestionDraft
        return QuestionDraft(
            question_text=revised_data.get("question_text", draft.question_text),
            question_type=revised_data.get("question_type", question_type),
            options=revised_data.get("options", draft.options),
            correct_answer=revised_data.get("correct_answer", draft.correct_answer),
            explanation=revised_data.get("explanation", draft.explanation),
            marks=revised_data.get("marks", marks),
            difficulty=revised_data.get("difficulty", difficulty),
            source_chunk_ids=revised_data.get("source_chunk_ids") or draft.source_chunk_ids or [],
            diagram_required=revised_data.get("diagram_required", diagram_required),
            diagram_spec=revised_data.get("diagram_spec", draft.diagram_spec),
            topic=revised_data.get("topic", draft.topic),
            bloom_level=revised_data.get("bloom_level", draft.bloom_level),
        )
    
    # =========================================================================
    # Full Generation Loop
    # =========================================================================
    
    async def generate_single_question(
        self,
        paper_id: str,
        section_index: int,
        slot_index: int,
        question_type: str,
        subject: str,
        grade: str,
        difficulty: str,
        marks: int,
        topic_focus: str,
        diagram_required: bool,
        tenant_id: str,
        pdf_ids: List[str],
        memory_state: PaperGenerationState,
    ) -> GeneratedQuestionRecord:
        """
        Generate a single question with the full two-LLM loop.
        
        ARCHITECTURE:
        1. LLM1 (Generator) creates question + diagram_spec
        2. LLM2 (Reviewer) checks quality and provides feedback
        3. If rejected, LLM1 revises using the feedback
        4. Repeat until approved or max iterations reached
        
        FEATURES:
        - Repetition detection: Detects when LLM1 outputs the same content
        - Hard correction mode: Forces changes when repetition detected
        - Comprehensive logging: Full traceability of each iteration
        
        Args:
            paper_id: Paper identifier
            section_index: Section index in the paper
            slot_index: Question slot index in the section
            question_type: Type of question (mcq, short, long, numerical)
            subject: Subject name
            grade: Grade level
            difficulty: Target difficulty (easy, medium, hard)
            marks: Marks for this question
            topic_focus: Topic to focus on
            diagram_required: Whether to include diagram
            tenant_id: Tenant identifier for RAG
            pdf_ids: Source PDF IDs for filtering
            memory_state: Current paper generation state
            
        Returns:
            GeneratedQuestionRecord with full generation history
        """
        start_time = time.time()
        
        # Initialize record
        record = GeneratedQuestionRecord(
            paper_id=paper_id,
            section_index=section_index,
            slot_index=slot_index,
            final_draft=QuestionDraft(
                question_text="",
                question_type=question_type,
                correct_answer="",
                explanation="",
                marks=marks,
                difficulty=difficulty,
            ),
        )
        
        # Initialize debug log for traceability
        debug_log = QuestionGenerationDebugLog(
            question_id=record.question_id,
            paper_id=paper_id,
            question_type=question_type,
            subject=subject,
            difficulty=difficulty,
            diagram_required=diagram_required,
        )
        
        # Track output hashes for repetition detection
        output_hashes: List[str] = []
        
        try:
            # ================================================================
            # STEP 1: Retrieve RAG context
            # ================================================================
            query_parts = [p for p in [subject, topic_focus, difficulty] if p]
            query = " ".join(query_parts) if query_parts else subject
            filters = {"source_file_id": pdf_ids} if pdf_ids else None
            
            logger.info(f"[Q{slot_index+1}] Starting generation: {question_type}, {subject}, {difficulty}, diagram={diagram_required}")
            
            rag_context, chunk_ids = await self.retrieve_context(
                query=query,
                tenant_id=tenant_id,
                filters=filters,
                top_k=8,
            )
            
            # Track if we're using LLM fallback (no RAG context)
            using_llm_fallback = False
            
            if not rag_context:
                logger.warning(f"[Q{slot_index+1}] No RAG context - using LLM knowledge fallback")
                using_llm_fallback = True
                debug_log.is_fallback_mode = True
                rag_context = [{
                    "text": f"Generate a {difficulty} difficulty {question_type} question about {topic_focus} for {subject}. Use your knowledge to create an appropriate question.",
                    "metadata": {"source": "llm_fallback"}
                }]
                chunk_ids = []
            
            # ================================================================
            # STEP 2: Generator-Reviewer Loop
            # ================================================================
            current_draft: Optional[QuestionDraft] = None
            current_review: Optional[ReviewResult] = None
            diagram_feedback: str = ""
            hard_correction_mode: bool = False
            
            for iteration in range(self.MAX_ITERATIONS):
                record.total_iterations = iteration + 1
                iteration_start = time.time()
                
                logger.info(f"[Q{slot_index+1}] === ITERATION {iteration + 1}/{self.MAX_ITERATIONS} ===")
                
                # --------------------------------------------------------------
                # LLM1: Generate or Revise
                # --------------------------------------------------------------
                generator_start = time.time()
                generator_prompt = ""
                
                if iteration == 0:
                    # Initial generation
                    logger.info(f"[Q{slot_index+1}] LLM1 (Generator): Creating initial question")
                    
                    context_str = format_rag_context(rag_context)
                    memory_context = memory_state.to_prompt_context()
                    generator_prompt = build_generator_prompt(
                        question_type=question_type,
                        subject=subject,
                        grade=grade,
                        difficulty=difficulty,
                        marks=marks,
                        topic_focus=topic_focus,
                        diagram_required=diagram_required,
                        rag_context=context_str,
                        chunk_ids=chunk_ids,
                        memory_context=memory_context,
                    )
                    
                    current_draft = await self.generate_question(
                        question_type=question_type,
                        subject=subject,
                        grade=grade,
                        difficulty=difficulty,
                        marks=marks,
                        topic_focus=topic_focus,
                        diagram_required=diagram_required,
                        rag_context=rag_context,
                        chunk_ids=chunk_ids,
                        memory_state=memory_state,
                    )
                else:
                    # Revision based on feedback
                    if hard_correction_mode:
                        logger.warning(f"[Q{slot_index+1}] LLM1 (Generator): HARD CORRECTION MODE - forcing different output")
                        # Use hard correction prompt
                        hard_prompt = self._build_hard_correction_prompt(
                            draft=current_draft,
                            review=current_review,
                            repetition_count=output_hashes.count(output_hashes[-1]) if output_hashes else 0,
                            diagram_feedback=diagram_feedback,
                        )
                        # Combine with revision prompt
                        generator_prompt = hard_prompt
                    else:
                        logger.info(f"[Q{slot_index+1}] LLM1 (Generator): Revising based on feedback")
                    
                    current_draft = await self.revise_question(
                        draft=current_draft,
                        review=current_review,
                        rag_context=rag_context,
                        question_type=question_type,
                        marks=marks,
                        difficulty=difficulty,
                        diagram_required=diagram_required,
                        subject=subject,
                        diagram_feedback=diagram_feedback,
                    )
                
                generator_duration_ms = int((time.time() - generator_start) * 1000)
                
                # Log LLM1 output
                logger.info(f"[Q{slot_index+1}] LLM1 output: '{current_draft.question_text[:100]}...'")
                if current_draft.diagram_required and current_draft.diagram_spec:
                    logger.info(f"[Q{slot_index+1}] LLM1 diagram_spec: type={current_draft.diagram_spec.get('diagram_type')}, params={list(current_draft.diagram_spec.get('params', {}).keys())}")
                
                # --------------------------------------------------------------
                # Repetition Detection
                # --------------------------------------------------------------
                current_hash = self._compute_content_hash(current_draft)
                is_repetition, repetition_count = self._detect_repetition(current_hash, output_hashes)
                output_hashes.append(current_hash)
                debug_log.output_hashes.append(current_hash)
                
                if is_repetition:
                    debug_log.repetition_count += 1
                    logger.warning(
                        f"[Q{slot_index+1}] REPETITION DETECTED! LLM1 produced same output as iteration "
                        f"{output_hashes.index(current_hash) + 1}. Enabling hard correction mode."
                    )
                    hard_correction_mode = True
                
                # --------------------------------------------------------------
                # Diagram Validation (before LLM2 review)
                # --------------------------------------------------------------
                diagram_feedback = ""
                diagram_validation_errors = []
                
                if current_draft.diagram_required and current_draft.diagram_spec:
                    diagram_validation = validate_diagram_spec(
                        current_draft.diagram_spec,
                        subject,
                        current_draft.question_text,
                    )
                    if not diagram_validation.is_valid:
                        diagram_validation_errors = diagram_validation.errors
                        logger.warning(f"[Q{slot_index+1}] Diagram validation FAILED: {diagram_validation.errors}")
                        diagram_feedback = build_diagram_revision_instructions(
                            diagram_validation,
                            subject,
                            current_draft.diagram_spec.get("diagram_type"),
                        )
                    elif diagram_validation.warnings:
                        logger.info(f"[Q{slot_index+1}] Diagram warnings: {diagram_validation.warnings}")
                        if diagram_validation.corrections:
                            spec = current_draft.diagram_spec or {}
                            params = spec.get("params", spec.get("parameters", {}))
                            for key, value in diagram_validation.corrections.items():
                                params[key] = value
                            spec["params"] = params
                            current_draft.diagram_spec = spec
                            logger.info(f"[Q{slot_index+1}] Applied automatic diagram corrections: {list(diagram_validation.corrections.keys())}")
                
                # --------------------------------------------------------------
                # LLM2: Review
                # --------------------------------------------------------------
                reviewer_start = time.time()
                logger.info(f"[Q{slot_index+1}] LLM2 (Reviewer): Evaluating question and diagram")
                
                current_review = await self.review_question(
                    draft=current_draft,
                    rag_context=rag_context,
                    chunk_ids=chunk_ids,
                    question_type=question_type,
                    target_difficulty=difficulty,
                    grade=grade,
                    is_fallback_mode=using_llm_fallback,
                )
                
                reviewer_duration_ms = int((time.time() - reviewer_start) * 1000)
                
                # Log LLM2 decision
                if current_review.approved:
                    logger.info(f"[Q{slot_index+1}] LLM2 verdict: APPROVED (quality_score={current_review.quality_score})")
                else:
                    issues_summary = ", ".join([f"{i.issue_type}: {i.message[:50]}" for i in current_review.issues[:3]])
                    logger.info(f"[Q{slot_index+1}] LLM2 verdict: REJECTED - [{issues_summary}]")
                    for issue in current_review.issues:
                        logger.info(f"[Q{slot_index+1}]   - [{issue.issue_type}] {issue.message}")
                        logger.info(f"[Q{slot_index+1}]     Fix: {issue.fix_instructions[:200]}")
                
                # Create iteration log
                iteration_log = self._create_iteration_log(
                    iteration=iteration + 1,
                    draft=current_draft,
                    review=current_review,
                    generator_prompt=generator_prompt,
                    generator_duration_ms=generator_duration_ms,
                    reviewer_duration_ms=reviewer_duration_ms,
                    is_revision=(iteration > 0),
                    is_hard_correction=hard_correction_mode,
                    repetition_detected=is_repetition,
                    diagram_validation_errors=diagram_validation_errors,
                )
                debug_log.iterations.append(iteration_log)
                
                # Record attempt
                attempt = QuestionGenerationAttempt(
                    attempt_number=iteration + 1,
                    draft=current_draft,
                    review=current_review,
                )
                record.attempts.append(attempt)
                
                # --------------------------------------------------------------
                # Check Approval Status
                # --------------------------------------------------------------
                diagram_approved = True
                if current_draft.diagram_required and diagram_feedback:
                    diagram_approved = False
                    logger.info(f"[Q{slot_index+1}] Diagram has issues - not approving yet")
                
                if current_review.approved and diagram_approved:
                    logger.info(f"[Q{slot_index+1}] ✅ QUESTION APPROVED after {iteration + 1} iterations")
                    record.approved = True
                    record.final_draft = current_draft
                    debug_log.final_approved = True
                    
                    if using_llm_fallback:
                        record.needs_manual_review = True
                        record.manual_review_reason = (
                            "Question generated without RAG context (source material not found). "
                            "Review to ensure accuracy and alignment with curriculum."
                        )
                    
                    # Learn from any issues for future questions
                    for issue in current_review.issues:
                        if issue.issue_type in ["hallucination", "concept_error"]:
                            memory_state.add_banned_pattern(f"Avoid: {issue.message[:100]}")
                    
                    break
                
                elif current_review.approved and not diagram_approved:
                    # Question text OK but diagram needs work
                    logger.info(f"[Q{slot_index+1}] Question text approved but diagram rejected")
                    current_review.issues.append(ReviewIssue(
                        issue_type="diagram",
                        message="Diagram specification has validation errors that must be fixed",
                        fix_instructions=diagram_feedback[:500] if diagram_feedback else "Fix diagram_spec parameters",
                    ))
                    current_review.approved = False
                
                # Learn from rejection
                for issue in current_review.issues:
                    if issue.issue_type in ["hallucination", "ambiguity"]:
                        memory_state.add_banned_pattern(issue.fix_instructions[:100])
                
                logger.info(f"[Q{slot_index+1}] --- End iteration {iteration + 1} (duration: {int((time.time() - iteration_start) * 1000)}ms) ---")
            
            # ================================================================
            # STEP 3: Final Status
            # ================================================================
            if not record.approved:
                logger.warning(f"[Q{slot_index+1}] ❌ QUESTION FAILED after {self.MAX_ITERATIONS} iterations")
                record.needs_manual_review = True
                record.manual_review_reason = (
                    f"Failed review after {self.MAX_ITERATIONS} iterations. "
                    f"Last issues: {[i.message for i in current_review.issues]}"
                )
                record.final_draft = current_draft
                debug_log.failure_reason = record.manual_review_reason
            
            record.completed_at = datetime.utcnow()
            debug_log.final_iteration_count = record.total_iterations
            debug_log.total_duration_ms = int((time.time() - start_time) * 1000)
            
            # Log final summary
            logger.info(
                f"[Q{slot_index+1}] Generation complete: approved={record.approved}, "
                f"iterations={record.total_iterations}, duration={debug_log.total_duration_ms}ms, "
                f"repetitions={debug_log.repetition_count}"
            )
            
            return record
            
        except Exception as e:
            logger.error(f"[Q{slot_index+1}] Generation EXCEPTION: {e}", exc_info=True)
            record.needs_manual_review = True
            record.manual_review_reason = f"Generation error: {str(e)}"
            record.completed_at = datetime.utcnow()
            debug_log.failure_reason = str(e)
            return record


# Singleton instance
_orchestrator: Optional[LLMOrchestrator] = None


async def get_orchestrator() -> LLMOrchestrator:
    """Get the singleton LLMOrchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = LLMOrchestrator()
        await _orchestrator.initialize()
    return _orchestrator
