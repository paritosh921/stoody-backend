"""ExamPen DCR API routes — all mounted under /api/v1/exampen/.

Mounting code (to be added to main_async.py after all route files are ready):

    from exampen.dcr.api.exam_orch import router as exampen_exam_router
    from exampen.dcr.api.stroke_ingest import router as exampen_stroke_router
    from exampen.dcr.api.score_engine import router as exampen_score_router
    from exampen.dcr.api.score_workflow import router as exampen_workflow_router
    from exampen.dcr.api.review import router as exampen_review_router
    from exampen.dcr.api.chat import router as exampen_chat_router
    from exampen.dcr.api.analytics import router as exampen_analytics_router
    from exampen.dcr.api.plagiarism import router as exampen_plagiarism_router
    from exampen.dcr.api.copy_upload import router as exampen_copy_upload_router
    from exampen.dcr.api.invig_console import router as exampen_invig_router
    from exampen.dcr.api.teacher_bff import router as exampen_teacher_bff_router
    from exampen.dcr.api.student_bff import router as exampen_student_bff_router

    EXAMPEN_PREFIX = f"{API_V1_PREFIX}/exampen"

    app.include_router(exampen_exam_router, prefix=f"{EXAMPEN_PREFIX}/exams", tags=["exampen-exams"])
    app.include_router(exampen_stroke_router, prefix=f"{EXAMPEN_PREFIX}/strokes", tags=["exampen-strokes"])
    app.include_router(exampen_score_router, prefix=f"{EXAMPEN_PREFIX}/scores", tags=["exampen-scores"])
    app.include_router(exampen_workflow_router, prefix=f"{EXAMPEN_PREFIX}/scores", tags=["exampen-workflow"])
    app.include_router(exampen_review_router, prefix=f"{EXAMPEN_PREFIX}/objections", tags=["exampen-review"])
    app.include_router(exampen_chat_router, prefix=f"{EXAMPEN_PREFIX}/chat", tags=["exampen-chat"])
    app.include_router(exampen_analytics_router, prefix=f"{EXAMPEN_PREFIX}/analytics", tags=["exampen-analytics"])
    app.include_router(exampen_plagiarism_router, prefix=f"{EXAMPEN_PREFIX}/plagiarism", tags=["exampen-plagiarism"])
    app.include_router(exampen_copy_upload_router, prefix=f"{EXAMPEN_PREFIX}/copies", tags=["exampen-copies"])
    app.include_router(exampen_invig_router, prefix=f"{EXAMPEN_PREFIX}/invig", tags=["exampen-invig"])
    app.include_router(exampen_teacher_bff_router, prefix=f"{EXAMPEN_PREFIX}/teacher", tags=["exampen-teacher-bff"])
    app.include_router(exampen_student_bff_router, prefix=f"{EXAMPEN_PREFIX}/student", tags=["exampen-student-bff"])
"""
