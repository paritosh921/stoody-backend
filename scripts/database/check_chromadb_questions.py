#!/usr/bin/env python3
"""
Check questions in ChromaDB
"""

from models import get_chromadb_client

def check_chromadb_questions():
    try:
        chromadb_client = get_chromadb_client()
        # Get all documents with metadata
        results = chromadb_client.collection.get(include=['metadatas', 'documents'])

        print(f"Total questions in ChromaDB: {len(results['ids'])}")

        # Group by document_type
        by_type = {}
        for i, metadata in enumerate(results['metadatas']):
            doc_type = metadata.get('document_type', 'unknown')
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append({
                'id': results['ids'][i],
                'document_id': metadata.get('document_id', 'unknown'),
                'subject': metadata.get('subject', 'unknown'),
                'difficulty': metadata.get('difficulty', 'unknown')
            })

        print("\nQuestions by document_type:")
        for doc_type, questions in by_type.items():
            print(f"  {doc_type}: {len(questions)} questions")
            # Show first few examples
            for q in questions[:3]:
                print(f"    - {q['id']}: {q['document_id']} ({q['subject']}, {q['difficulty']})")
            if len(questions) > 3:
                print(f"    ... and {len(questions) - 3} more")

    except Exception as e:
        print(f"Error checking ChromaDB: {e}")

if __name__ == '__main__':
    check_chromadb_questions()
