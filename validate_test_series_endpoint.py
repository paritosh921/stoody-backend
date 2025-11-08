# This code should be added to backend/api/v1/admin_async.py at the end of the file

@router.post("/validate-test-series/{document_id}")
@limiter.limit("30/minute")
async def validate_test_series(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Validate test series before making it available to students:
    1. Total points matches sum of question points
    2. Total minutes > 0
    3. All questions have correct_answer set
    """
    try:
        # Get document
        document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if document.get("document_type") != "Test Series":
            raise HTTPException(status_code=400, detail="Only Test Series can be validated")

        # Get questions
        questions = await db.mongo_find("questions", {"document_id": document_id})

        if not questions or len(questions) == 0:
            raise HTTPException(
                status_code=400,
                detail="No questions found for this test series. Please process OCR first."
            )

        # Validation 1: Total points
        total_allocated = sum(q.get("points", 1) for q in questions)
        total_points = document.get("total_points", 0)
        if abs(total_allocated - total_points) >= 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"Total points ({total_points}) doesn't match allocated points ({total_allocated})"
            )

        # Validation 2: Total minutes
        total_minutes = document.get("total_minutes", 0)
        if not total_minutes or total_minutes <= 0:
            raise HTTPException(status_code=400, detail="Total minutes must be greater than 0")

        # Validation 3: All questions have correct_answer
        questions_without_answer = [q for q in questions if not q.get("correct_answer") or q.get("correct_answer", "").strip() == ""]
        if questions_without_answer:
            raise HTTPException(
                status_code=400,
                detail=f"{len(questions_without_answer)} question(s) don't have correct answer set"
            )

        # Mark as validated
        await db.mongo_update_one(
            "documents",
            {"document_id": document_id},
            {"$set": {"is_validated": True, "validated_at": datetime.utcnow()}}
        )

        logger.info(f"Test series {document_id} validated successfully")

        return {
            "success": True,
            "message": "Test series validated successfully",
            "data": {
                "total_questions": len(questions),
                "total_points": total_points,
                "total_minutes": total_minutes
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate test series: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
