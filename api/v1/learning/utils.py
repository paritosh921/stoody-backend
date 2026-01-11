def grades_match(student_grade: str, doc_standard: str) -> bool:
    """
    Check if student grade matches document standard.
    Uses flexible matching to handle various grade formats.
    
    Args:
        student_grade: Student's grade from profile (e.g., "12", "12th", "Class 12")
        doc_standard: Document's standard field (e.g., "12", "12th Pass")
    
    Returns:
        True if grades match, False otherwise
    """
    if not student_grade or not doc_standard:
        return False
    
    # Exact match first
    if student_grade == doc_standard:
        return True
    
    # Normalize both values for comparison
    def normalize_grade(grade: str) -> str:
        """Normalize grade to just the number"""
        if not grade:
            return ""
        grade = str(grade).lower().strip()
        # Remove common suffixes
        for suffix in ["th", "st", "nd", "rd", " pass", " class", "class "]:
            grade = grade.replace(suffix, "")
        return grade.strip()
    
    normalized_student = normalize_grade(student_grade)
    normalized_doc = normalize_grade(doc_standard)
    
    return normalized_student == normalized_doc
