"""
Diagram Pipeline Service

Orchestrates the complete diagram generation pipeline:
1. Question → LLM1 instructions
2. Instructions → Kimi-authored TikZ image
3. Image + instructions → LLM2 review
4. Review → improved instructions (if needed)

Loop continues until image is acceptable or max iterations reached.
"""

import logging
import time
import os
from typing import Optional, Dict, Any, Union
from datetime import datetime

from .models import (
    DiagramGenerationRequest,
    DiagramGenerationResponse,
    DiagramInstructions,
    DiagramReview,
    DiagramImage,
    QuestionDiagramRecord,
    SubjectType,
    ReviewIssue,
    IssueCode,
)
from .kimi_client import LLM1InstructionGenerator, LLM2DiagramReviewer
from .tikz_client import TikzImageClient

logger = logging.getLogger(__name__)


class DiagramPipelineService:
    """
    Main service for the diagram generation pipeline.
    
    Handles the complete flow from question text to final diagram image,
    including iterative refinement based on LLM2 reviews.
    """
    
    def __init__(self, db=None):
        """
        Initialize the pipeline service.
        
        Args:
            db: Optional DatabaseManager or MongoDB database for storing diagram records
        """
        self.db_manager = db
        self.llm1 = LLM1InstructionGenerator()
        self.llm2 = LLM2DiagramReviewer()
        self.image_generator = TikzImageClient()
        
        # Collection name for diagram records
        self.collection_name = "question_diagrams"
        self.image_review_enabled = os.getenv("DIAGRAM_IMAGE_REVIEW_ENABLED", "true").lower() not in ["0", "false", "no"]
        # Strict mode rejects diagrams when the rendered image description
        # misses required labels/values or adds contradictory ones.
        self.strict_image_consistency = os.getenv("DIAGRAM_STRICT_IMAGE_CONSISTENCY", "true").lower() in ["1", "true", "yes"]
        # Strict instruction coverage rejects before rendering when labels/values are missing.
        self.strict_instruction_coverage = os.getenv("DIAGRAM_STRICT_INSTRUCTION_COVERAGE", "false").lower() in ["1", "true", "yes"]

    def _instruction_coverage_issues(self, question_text: str, instructions: str) -> Dict[str, Any]:
        """
        Check that instructions include all labels and numeric values from the question.

        Returns a dict with missing labels and numbers for feedback.
        """
        q_meta = self._extract_labels_numbers(question_text, exclude_quantity_labels=True)
        i_meta = self._extract_labels_numbers(instructions, exclude_quantity_labels=True)

        q_labels = q_meta["labels"]
        i_labels = i_meta["labels"]
        q_numbers = q_meta["numbers"]
        i_numbers = i_meta["numbers"]

        missing_labels = sorted(q_labels - i_labels)

        # Numeric coverage should be strong but not absolute.
        # Require full match for tiny sets; otherwise require >= ceil(2n/3).
        matched_numbers = q_numbers & i_numbers
        if len(q_numbers) <= 2:
            required_matches = len(q_numbers)
        else:
            required_matches = (2 * len(q_numbers) + 2) // 3
        missing_numbers = sorted(q_numbers - i_numbers) if len(matched_numbers) < required_matches else []

        return {
            "missing_labels": missing_labels,
            "missing_numbers": missing_numbers,
        }

    def _normalize_visual_text(self, text: str) -> str:
        """Normalize common Unicode symbols so regex extraction is stable."""
        if not text:
            return ""
        subscript_map = str.maketrans({
            "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
            "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
            "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
            "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
            "−": "-", "–": "-", "—": "-",
            "°": " deg ",
        })
        return (text or "").translate(subscript_map)

    def _extract_labels_numbers(self, text: str, exclude_quantity_labels: bool = False) -> Dict[str, Any]:
        import re

        normalized = self._normalize_visual_text(text)
        labels = set()
        quantity_labels = set()

        # 1) Sequence labels like triangle ABC, line PQ, chord AB
        seq_patterns = [
            r"\b(?:(?i:triangle|line|segment|chord|arc|angle|quadrilateral|polygon|points?|vertices|vertex))\s+([A-Z]{2,6})\b",
            r"\b(?:(?i:between|join(?:ing)?|from|to))\s+([A-Z]{2,6})\b",
        ]
        for pat in seq_patterns:
            for seq in re.findall(pat, normalized or ""):
                labels.update(list(seq.upper()))

        # 2) Explicit single-label mentions (point A, vertex B, block B, label C)
        single_patterns = [
            r"\b(?:(?i:point|vertex|label(?:ed)?|block|particle|node|terminal))\s+([A-Z])\b",
            r"\b([A-Z])\s*=",
            r"\b([A-Z])\s*\(",
        ]
        for pat in single_patterns:
            for lbl in re.findall(pat, normalized or ""):
                labels.add(lbl.upper())

        # Quantity symbols like mass M, radius R, charge Q are optional in many diagrams.
        quantity_patterns = [
            r"\b(?i:mass|radius|length|height|distance|separation|diameter|charge|current|voltage|resistance|speed|velocity|acceleration|force|power|energy|time|frequency|wavelength|moment|inertia|temperature|pressure|coefficient)\s+([A-Z])\b",
        ]
        for pat in quantity_patterns:
            for lbl in re.findall(pat, normalized or ""):
                quantity_labels.add(lbl.upper())

        # 3) Fallback for pure geometry shorthand like "ABC is a triangle"
        for seq in re.findall(r'\b([A-Z]{2,6})\b', normalized or ""):
            seq = seq.upper()
            if seq in {"JEE", "NEET", "MCQ", "N", "KG", "CM", "MM", "MS"}:
                continue
            labels.update(list(seq))

        numbers = set(re.findall(r'\d+(?:\.\d+)?', normalized or ""))
        if exclude_quantity_labels and quantity_labels:
            labels = labels - quantity_labels
        return {"labels": labels, "numbers": numbers}
    
    async def generate_diagram(
        self,
        request: DiagramGenerationRequest,
    ) -> DiagramGenerationResponse:
        """
        Generate a diagram for a question using the full pipeline.
        
        Pipeline flow:
        1. LLM1 generates instructions
        2. Nano Banan Pro generates image
        3. LLM2 reviews the result
        4. If not acceptable, refine and repeat (up to max_iterations)
        
        Args:
            request: DiagramGenerationRequest with question details
        
        Returns:
            DiagramGenerationResponse with final image and history
        """
        start_time = time.time()
        
        instructions_history = []
        review_history = []
        current_image = None
        accepted_image = None
        last_review = None
        
        try:
            # Check for existing diagram record
            existing_record = await self._get_diagram_record(request.question_id)
            
            if existing_record and not request.force_regenerate:
                # Return existing diagram if it's finalized
                if existing_record.is_finalized and existing_record.current_image:
                    return DiagramGenerationResponse(
                        question_id=request.question_id,
                        success=True,
                        image=existing_record.current_image,
                        instructions_history=existing_record.instructions_history,
                        review_history=existing_record.review_history,
                        iterations_used=0,
                        total_time_seconds=time.time() - start_time,
                    )
            
            # Start the pipeline
            current_instructions = None
            previous_review = None

            required_meta = self._extract_labels_numbers(request.question_text)
            required_labels = sorted(required_meta["labels"])
            required_numbers = sorted(required_meta["numbers"])

            for iteration in range(request.max_iterations):
                logger.info(f"Diagram pipeline iteration {iteration + 1} for question {request.question_id}")
                
                # Step 1: Generate instructions (LLM1)
                if current_instructions is None:
                    # First iteration - generate fresh instructions
                    instructions = await self.llm1.generate_instructions(
                        question_text=request.question_text,
                        subject=request.subject,
                        diagram_hints=request.diagram_hints,
                        required_labels=required_labels,
                        required_numbers=required_numbers,
                    )
                else:
                    # Refinement iteration - improve based on review
                    review_feedback = None
                    if previous_review and previous_review.suggested_instruction_update:
                        review_feedback = previous_review.suggested_instruction_update
                    elif previous_review and previous_review.issues:
                        review_feedback = "; ".join([f"{i.code}: {i.details}" for i in previous_review.issues])
                    
                    instructions = await self.llm1.generate_instructions(
                        question_text=request.question_text,
                        subject=request.subject,
                        diagram_hints=request.diagram_hints,
                        previous_instructions=current_instructions.instructions,
                        review_feedback=review_feedback,
                        required_labels=required_labels,
                        required_numbers=required_numbers,
                    )
                
                # Update version number
                instructions.version = iteration + 1
                if iteration > 0:
                    instructions.refined_from_version = iteration
                
                instructions_history.append(instructions)
                current_instructions = instructions

                # Pre-check: ensure instructions cover all labels/values from the question
                coverage = self._instruction_coverage_issues(
                    request.question_text,
                    instructions.instructions,
                )
                if coverage["missing_labels"] or coverage["missing_numbers"]:
                    if not self.strict_instruction_coverage:
                        logger.info(
                            "Instruction coverage warnings; proceeding without blocking. "
                            f"Missing labels={coverage['missing_labels']}, numbers={coverage['missing_numbers']}"
                        )
                    else:
                        issues = []
                        if coverage["missing_labels"]:
                            issues.append(ReviewIssue(
                                code=IssueCode.MISSING_LABEL,
                                details=f"Missing labels: {', '.join(coverage['missing_labels'])}",
                            ))
                        if coverage["missing_numbers"]:
                            issues.append(ReviewIssue(
                                code=IssueCode.INCOMPLETE_DIAGRAM,
                                details=f"Missing values: {', '.join(coverage['missing_numbers'])}",
                            ))

                        feedback_parts = []
                        if coverage["missing_labels"]:
                            feedback_parts.append(f"Add labels: {', '.join(coverage['missing_labels'])}.")
                        if coverage["missing_numbers"]:
                            feedback_parts.append(f"Include values: {', '.join(coverage['missing_numbers'])}.")

                        review = DiagramReview(
                            is_acceptable=False,
                            issues=issues,
                            suggested_instruction_update=" ".join(feedback_parts).strip() or None,
                        )
                        review.review_version = iteration + 1
                        review_history.append(review)
                        previous_review = review
                        logger.info(
                            f"Instruction coverage failed; refining. "
                            f"Missing labels={coverage['missing_labels']}, numbers={coverage['missing_numbers']}"
                        )
                        continue

                # Step 2: Generate image (Nano Banan Pro)
                try:
                    image = await self.image_generator.generate_diagram(
                        instructions=instructions.instructions,
                        question_id=request.question_id,
                        instructions_version=instructions.version,
                    )
                    current_image = image
                except Exception as e:
                    logger.error(f"Image generation failed: {e!r}")
                    logger.exception("Image generation exception stack trace")
                    # Continue to try with refined instructions
                    continue
                
                # Step 3: Review (LLM2) with optional image description
                image_description = None
                if self.image_review_enabled and current_image and current_image.base64_data:
                    try:
                        image_description = await self.image_generator.describe_diagram(
                            current_image.base64_data,
                            question_text=request.question_text,
                            instructions=instructions.instructions,
                        )
                        logger.info(
                            f"Image description length: {len(image_description) if image_description else 0}"
                        )
                    except Exception as desc_error:
                        logger.warning(f"Image description failed: {desc_error}")

                if image_description and self.strict_image_consistency:
                    instr_meta = self._extract_labels_numbers(instructions.instructions)
                    i_meta = self._extract_labels_numbers(image_description)

                    missing_labels = sorted(instr_meta["labels"] - i_meta["labels"])
                    missing_numbers = sorted(instr_meta["numbers"] - i_meta["numbers"])
                    extra_labels = sorted(i_meta["labels"] - instr_meta["labels"])
                    extra_numbers = sorted(i_meta["numbers"] - instr_meta["numbers"])

                    # Hard-reject only when required instruction elements are missing.
                    # Extra labels/values are logged and then left to LLM2 for semantic judgment.
                    if missing_labels or missing_numbers:
                        issues = []
                        if missing_labels:
                            issues.append(ReviewIssue(
                                code=IssueCode.MISSING_LABEL,
                                details=f"Image is missing required labels: {', '.join(missing_labels)}",
                            ))
                        if missing_numbers:
                            issues.append(ReviewIssue(
                                code=IssueCode.INCOMPLETE_DIAGRAM,
                                details=f"Image is missing required values: {', '.join(missing_numbers)}",
                            ))

                        feedback_parts = []
                        if missing_labels:
                            feedback_parts.append(f"Ensure labels are visible: {', '.join(missing_labels)}.")
                        if missing_numbers:
                            feedback_parts.append(f"Ensure values are visible: {', '.join(missing_numbers)}.")

                        review = DiagramReview(
                            is_acceptable=False,
                            issues=issues,
                            suggested_instruction_update=" ".join(feedback_parts).strip() or None,
                        )
                        review.review_version = iteration + 1
                        review_history.append(review)
                        previous_review = review
                        logger.info(
                            f"Image missing required elements; refining. "
                            f"Missing labels={missing_labels}, missing numbers={missing_numbers}"
                        )
                        continue
                    elif extra_labels or extra_numbers:
                        logger.info(
                            f"Image has extra labels/values (soft warning). "
                            f"extra labels={extra_labels}, extra numbers={extra_numbers}"
                        )

                review = await self.llm2.review_diagram(
                    question_text=request.question_text,
                    current_instructions=instructions.instructions,
                    subject=request.subject,
                    image_description=image_description,
                )

                # Post-process review to reduce false negatives on generic physics/math prompts.
                import re
                required_meta = self._extract_labels_numbers(request.question_text, exclude_quantity_labels=True)
                required_labels = required_meta["labels"]
                q_lower = (request.question_text or "").lower()
                arrow_required = any(
                    re.search(p, q_lower)
                    for p in [
                        r"\bforce\b", r"\bvelocity\b", r"\bacceleration\b",
                        r"\bcurrent\b", r"\bfield\b", r"\bray\b", r"\bdirection\b",
                        r"\bvector\b",
                    ]
                )

                filtered_issues = []
                for issue in review.issues:
                    if issue.code == IssueCode.MISSING_LABEL and not required_labels:
                        logger.info("Downgrading MISSING_LABEL: no explicit labels required by question")
                        continue
                    if issue.code == IssueCode.MISSING_ARROW and not arrow_required:
                        logger.info("Downgrading MISSING_ARROW: no directional arrow requirement detected")
                        continue
                    filtered_issues.append(issue)
                review.issues = filtered_issues

                soft_issue_codes = {
                    IssueCode.STYLE_ISSUE,
                    IssueCode.LABEL_OVERLAP,
                    IssueCode.LOW_CLARITY,
                    IssueCode.MISSING_ARROW,
                    IssueCode.WRONG_CONFIGURATION,
                    IssueCode.WRONG_PROPORTION,
                }
                if review.issues and all(i.code in soft_issue_codes for i in review.issues):
                    logger.info(
                        "Accepting diagram with only soft review issues: "
                        f"{[i.code.value for i in review.issues]}"
                    )
                    review.is_acceptable = True
                elif not review.issues:
                    review.is_acceptable = True

                logger.info(
                    f"Diagram review (iter {iteration + 1}): acceptable={review.is_acceptable}, "
                    f"issues={[i.code.value for i in review.issues]}"
                )
                review.review_version = iteration + 1
                review_history.append(review)
                previous_review = review
                last_review = review
                
                # Check if acceptable
                if review.is_acceptable:
                    logger.info(f"Diagram accepted for question {request.question_id} after {iteration + 1} iterations")
                    accepted_image = current_image
                    break
                
                logger.info(f"Diagram needs refinement: {[i.code for i in review.issues]}")
            
            # Save the record
            fallback_enabled = os.getenv("DIAGRAM_FALLBACK_ACCEPT", "false").lower() not in ["0", "false", "no"]
            fallback_allowed_issues = {"STYLE_ISSUE", "LABEL_OVERLAP", "LOW_CLARITY"}
            if accepted_image is None and fallback_enabled and current_image is not None:
                # Accept a fallback image only if review issues are non-contradictory
                issue_codes = set()
                if last_review:
                    issue_codes = {i.code.value for i in last_review.issues}
                if issue_codes and not issue_codes.issubset(fallback_allowed_issues):
                    logger.warning(
                        f"Fallback rejected for question {request.question_id}; "
                        f"issue_codes={sorted(issue_codes)}"
                    )
                else:
                    logger.warning(
                        f"Accepting fallback diagram for question {request.question_id}; "
                        f"last_review_acceptable={last_review.is_acceptable if last_review else None}"
                    )
                    accepted_image = current_image

            record = QuestionDiagramRecord(
                question_id=request.question_id,
                current_image=accepted_image,
                instructions_history=instructions_history,
                review_history=review_history,
                generation_count=(existing_record.generation_count + 1) if existing_record else 1,
                is_finalized=review_history[-1].is_acceptable if review_history else False,
                needs_review=(accepted_image is None) or (last_review is not None and not last_review.is_acceptable),
                updated_at=datetime.utcnow(),
            )
            
            await self._save_diagram_record(record)
            
            return DiagramGenerationResponse(
                question_id=request.question_id,
                success=accepted_image is not None,
                image=accepted_image,
                instructions_history=instructions_history,
                review_history=review_history,
                iterations_used=len(instructions_history),
                total_time_seconds=time.time() - start_time,
                error_message=None if accepted_image is not None else "No acceptable diagram produced",
                error_code=None if accepted_image is not None else "NO_ACCEPTABLE_DIAGRAM",
            )
            
        except Exception as e:
            logger.error(f"Diagram pipeline failed for question {request.question_id}: {e}")
            
            return DiagramGenerationResponse(
                question_id=request.question_id,
                success=False,
                instructions_history=instructions_history,
                review_history=review_history,
                iterations_used=len(instructions_history),
                total_time_seconds=time.time() - start_time,
                error_message=str(e),
                error_code="PIPELINE_ERROR",
            )
    
    async def generate_instructions_only(
        self,
        question_text: str,
        subject: Optional[SubjectType] = None,
        diagram_hints: Optional[str] = None,
    ) -> DiagramInstructions:
        """
        Generate only the diagram instructions without image generation.
        
        Useful for previewing or manual refinement.
        
        Args:
            question_text: The question text
            subject: Optional subject type
            diagram_hints: Optional hints
        
        Returns:
            DiagramInstructions object
        """
        return await self.llm1.generate_instructions(
            question_text=question_text,
            subject=subject,
            diagram_hints=diagram_hints,
        )
    
    async def generate_image_from_instructions(
        self,
        instructions: str,
        question_id: str,
        version: int = 1,
    ) -> DiagramImage:
        """
        Generate an image from provided instructions.
        
        Useful when instructions are manually edited.
        
        Args:
            instructions: The diagram instructions
            question_id: Question ID for file naming
            version: Version number
        
        Returns:
            DiagramImage object
        """
        return await self.image_generator.generate_diagram(
            instructions=instructions,
            question_id=question_id,
            instructions_version=version,
        )
    
    async def review_instructions(
        self,
        question_text: str,
        instructions: str,
        subject: Optional[SubjectType] = None,
    ) -> DiagramReview:
        """
        Review diagram instructions without generating an image.
        
        Useful for validating manually written instructions.
        
        Args:
            question_text: The question text
            instructions: The instructions to review
            subject: Optional subject type
        
        Returns:
            DiagramReview object
        """
        return await self.llm2.review_diagram(
            question_text=question_text,
            current_instructions=instructions,
            subject=subject,
        )
    
    async def get_diagram_record(
        self,
        question_id: str,
    ) -> Optional[QuestionDiagramRecord]:
        """
        Get the diagram record for a question.
        
        Args:
            question_id: The question ID
        
        Returns:
            QuestionDiagramRecord if exists, None otherwise
        """
        return await self._get_diagram_record(question_id)
    
    async def _get_diagram_record(
        self,
        question_id: str,
    ) -> Optional[QuestionDiagramRecord]:
        """Internal method to get diagram record from database."""
        if not self.db_manager:
            return None
        
        try:
            doc = await self.db_manager.mongo_find_one(
                self.collection_name,
                {"question_id": question_id}
            )
            
            if doc:
                # Convert MongoDB document to model
                doc.pop("_id", None)
                return QuestionDiagramRecord(**doc)
        except Exception as e:
            logger.error(f"Failed to get diagram record: {e}")
        
        return None
    
    async def _save_diagram_record(
        self,
        record: QuestionDiagramRecord,
    ) -> None:
        """Internal method to save diagram record to database."""
        if not self.db_manager:
            return
        
        try:
            # Convert to dict for MongoDB
            doc = record.model_dump()
            
            # Convert datetime objects to strings for MongoDB
            def convert_dates(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_dates(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_dates(item) for item in obj]
                return obj
            
            doc = convert_dates(doc)
            
            # Upsert the record
            await self.db_manager.mongo_update_one(
                self.collection_name,
                {"question_id": record.question_id},
                {"$set": doc},
                upsert=True,
            )
        except Exception as e:
            logger.error(f"Failed to save diagram record: {e}")
    
    async def delete_diagram(
        self,
        question_id: str,
    ) -> bool:
        """
        Delete a diagram and its record.
        
        Args:
            question_id: The question ID
        
        Returns:
            True if deleted, False if not found
        """
        record = await self._get_diagram_record(question_id)
        
        if record and record.current_image:
            # Delete the image file
            self.image_generator.delete_image(record.current_image.path)
        
        if self.db_manager:
            try:
                result = await self.db_manager.mongo_delete_one(
                    self.collection_name,
                    {"question_id": question_id}
                )
                return result.deleted_count > 0
            except Exception as e:
                logger.error(f"Failed to delete diagram record: {e}")
        
        return False
