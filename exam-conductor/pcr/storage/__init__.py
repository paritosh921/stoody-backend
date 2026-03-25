"""
PCR Storage Layer
=================

Async MongoDB repository layer for all PCR conducted-exam collections.

This package owns persistence for:
- ``evalpen_detected_responses`` — segmented student responses with flags
- ``evalpen_evaluations`` — LLM evaluation results with audit trail
- ``evalpen_questions`` — question metadata with complexity and eval template
- ``evalpen_solutions`` — versioned reference solutions

It reads from (but does NOT write to):
- ``evalpen_submissions`` — canonical submissions (owned by ingest substrate)
- ``evalpen_answer_pages`` — canonical page artifacts (owned by ingest substrate)

Ownership Declaration
---------------------
- Writes: evalpen_detected_responses, evalpen_evaluations,
  evalpen_questions, evalpen_solutions
- Reads from: evalpen_submissions, evalpen_answer_pages
  (owned by ingest substrate)
- Never writes to: evalpen_submissions (raw artifact data),
  evalpen_answer_pages, practice collections
- Transactional boundaries: per-collection atomic writes with audit

Integrity rules
---------------
- Detected responses are immutable after write (TAMPER_PROOF_SPEC Layer 1)
- Evaluations include append-only audit trail (TAMPER_PROOF_SPEC Layer 3)
- Solutions are versioned and append-only at the version level
- Questions are mutable (teacher-uploaded content)

References
----------
- Architecture: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md (Section 7)
- Integrity:    new-docs/architecture/TAMPER_PROOF_SPEC.md
- Ownership:    new-docs/governance/STATE_OWNERSHIP_MAP.md
- Failure modes: PCR-01, PCR-02, PCR-03, PCR-04
- Test IDs:     U-EVAL-01, I-PCR-01, I-PCR-02, I-PCR-03,
                I-TAMP-02, I-TAMP-03
"""

from .submission_repo import SubmissionRepository
from .response_repo import (
    DetectedResponseRepository,
    ImmutableResponseError,
)
from .evaluation_repo import (
    EvaluationRepository,
    DuplicateEvaluationError,
)
from .question_repo import QuestionRepository
from .solution_repo import (
    SolutionRepository,
    SolutionVersionConflictError,
)

__all__ = [
    # Repositories
    "SubmissionRepository",
    "DetectedResponseRepository",
    "EvaluationRepository",
    "QuestionRepository",
    "SolutionRepository",
    # Errors
    "ImmutableResponseError",
    "DuplicateEvaluationError",
    "SolutionVersionConflictError",
]
