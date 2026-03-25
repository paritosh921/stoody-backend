"""
PCR Services Layer
==================

High-level orchestration for PCR evaluation: OCR adapter, submission
processing, solution cache, and evaluation core.

This package sits between the PCR domain/storage layers and the route
handlers.  All LLM-mediated work is routed through the shared LLM gate
(C4).

Ownership Declaration
---------------------
- Writes: evalpen_detected_responses (eval_status), evalpen_evaluations,
  evalpen_solutions (cache warmup)
- Reads from: evalpen_submissions (via ingest), evalpen_answer_pages,
  evalpen_detected_responses, evalpen_questions, evalpen_solutions,
  llm_gate (via LLMGate)
- Never writes to: evalpen_submissions (raw artifact data),
  evalpen_answer_pages, practice persistence
- Transactional boundaries: evaluation output + audit + gate usage refs

References
----------
- Architecture: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md (Sections 3-5)
- Gate:         new-docs/architecture/LLM_GATE_SPEC.md
- Integrity:    new-docs/architecture/TAMPER_PROOF_SPEC.md (Layer 2, Layer 3)
- Failure modes: PCR-01, PCR-02, PCR-03, PCR-04
- Test IDs:     U-EVAL-01, I-PCR-01, I-PCR-02, I-PCR-03
"""

from .ocr_service import OCRAdapter, OCRResult, LLMVisionCameraAdapter, LLMVisionPenAdapter, VisionGateProtocol, create_ocr_adapter
from .solution_cache import SolutionCache
from .submission_service import SubmissionService
from .eval_core import EvalCore

__all__ = [
    # OCR
    "OCRAdapter",
    "OCRResult",
    "LLMVisionCameraAdapter",
    "LLMVisionPenAdapter",
    "VisionGateProtocol",
    "create_ocr_adapter",
    # Solution cache
    "SolutionCache",
    # Submission orchestration
    "SubmissionService",
    # Evaluation core
    "EvalCore",
]
