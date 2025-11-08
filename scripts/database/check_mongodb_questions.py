#!/usr/bin/env python3
"""
Check questions in MongoDB
"""

from models import get_collection

def check_mongodb_questions():
    questions_collection = get_collection('questions')

    # Get all questions
    questions = list(questions_collection.find({}, {
        'id': 1,
        'text': 1,
        'document_id': 1,
        'document_type': 1,
        'subject': 1,
        'difficulty': 1
    }))

    print(f"Total questions in MongoDB: {len(questions)}")

    # Group by document_id and document_type
    by_doc = {}
    by_type = {}
    for q in questions:
        doc_id = q.get('document_id', 'unknown')
        doc_type = q.get('document_type', 'unknown')

        if doc_id not in by_doc:
            by_doc[doc_id] = []
        by_doc[doc_id].append(q)

        if doc_type not in by_type:
            by_type[doc_type] = []
        by_type[doc_type].append(q)

    print("\nQuestions by document_id:")
    for doc_id, qlist in by_doc.items():
        print(f"  {doc_id}: {len(qlist)} questions")
        for q in qlist[:2]:  # Show first 2
            print(f"    - {q['id']}: {q.get('subject', 'unknown')} ({q.get('difficulty', 'unknown')})")
        if len(qlist) > 2:
            print(f"    ... and {len(qlist) - 2} more")

    print("\nQuestions by document_type:")
    for doc_type, qlist in by_type.items():
        print(f"  {doc_type}: {len(qlist)} questions")

if __name__ == '__main__':
    check_mongodb_questions()
