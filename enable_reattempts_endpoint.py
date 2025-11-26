
@router.post("/admin/enable-all-reattempts")
async def enable_all_reattempts(
    current_user: Dict[str, Any] = Depends(require_admin_for_write),
    db: DatabaseManager = Depends(get_database)
):
    """
    Admin endpoint to enable re-attempts for all existing test attempts
    This is a one-time fix for the migration
    """
    try:
        # Update all existing attempts to allow re-attempts
        from pymongo import UpdateMany
        
        result = await db.student_test_attempts.update_many(
            {},  # Match all documents
            {"$set": {"can_reattempt": True}}
        )
        
        logger.info(f"Admin {current_user['user_id']} enabled re-attempts for {result.modified_count} test attempts")
        
        return {
            "success": True,
            "message": f"Updated {result.modified_count} test attempts to allow re-attempts",
            "matched_count": result.matched_count,
            "modified_count": result.modified_count
        }
        
    except Exception as e:
        logger.error(f"Enable re-attempts error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enable re-attempts: {str(e)}"
        )
