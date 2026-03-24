"""NATS subject constants for the ExamPen DCR event bus.

All subjects live under the ``EXAMPEN.`` prefix so they are captured
by a single JetStream stream (see :mod:`stream_setup`).
"""

# -- Stroke pipeline ----------------------------------------------------------
STROKE_RAW = "EXAMPEN.stroke.raw"
STROKE_PROCESSED = "EXAMPEN.stroke.processed"

# -- Page assembly ------------------------------------------------------------
PAGE_READY = "EXAMPEN.page.ready"

# -- AI inference & scoring ---------------------------------------------------
AI_RESULT = "EXAMPEN.ai.result"
SCORE_UPDATED = "EXAMPEN.score.updated"
RESCORE_COMMAND = "EXAMPEN.score.rescore_command"

# -- Exam lifecycle -----------------------------------------------------------
EXAM_LIFECYCLE = "EXAMPEN.exam.lifecycle"

# -- Objections ---------------------------------------------------------------
OBJECTION = "EXAMPEN.objection"

# -- Plagiarism ---------------------------------------------------------------
PLAGIARISM_CHECK = "EXAMPEN.plagiarism.check"
PLAGIARISM_RESULT = "EXAMPEN.plagiarism.result"

# -- Copy readiness -----------------------------------------------------------
COPY_READY = "EXAMPEN.copy.ready"
