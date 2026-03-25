"""
ExamPen Shared Ingest Substrate
===============================

Canonical conducted-exam artifact persistence used by both DCR and PCR engines.

Ownership Declaration
---------------------
- Writes: evalpen_submissions, evalpen_answer_pages
- Reads from: exam metadata (exam_type, roster) owned by orchestration layer
- Never writes to: practice collections, DCR result collections, PCR result collections
- Transactional boundaries: pages persisted before submission; orphan cleanup on failure (best-effort, not a true multi-document transaction)

References
----------
- Architecture: new-docs/architecture/DUAL_MODE_ARCHITECTURE.md  (Section 3)
- Integrity:    new-docs/architecture/TAMPER_PROOF_SPEC.md       (Layer 1)
- Ownership:    new-docs/governance/STATE_OWNERSHIP_MAP.md
- Failure modes: ING-01, ING-02, ING-03
- Test IDs:     U-ING-01, U-ING-02, I-ING-01, I-ING-02
"""

from .models import (
    ArtifactSource,
    SubmissionStatus,
    PageRef,
    ConductedExamSubmission,
    AnswerPage,
    IngestResult,
)
from .hashing import compute_content_hash, compute_page_hash
from .repository import IngestRepository
from .service import IngestService

__all__ = [
    # Models
    "ArtifactSource",
    "SubmissionStatus",
    "PageRef",
    "ConductedExamSubmission",
    "AnswerPage",
    "IngestResult",
    # Hashing
    "compute_content_hash",
    "compute_page_hash",
    # Repository
    "IngestRepository",
    # Service
    "IngestService",
]
