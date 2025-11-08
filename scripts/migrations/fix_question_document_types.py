#!/usr/bin/env python3
"""
Fix document_type field for existing questions in MongoDB
"""

from models import get_collection, get_chromadb_client

def fix_question_document_types():
    """Update existing questions with correct document_type based on their document_id"""

    questions_collection = get_collection('questions')
    documents_collection = get_collection('documents')
    chromadb_client = get_chromadb_client()

    # Get all questions that have document_id but missing or unknown document_type
    questions_to_fix = list(questions_collection.find({
        'document_id': {'$exists': True, '$ne': None},
        '$or': [
            {'document_type': {'$exists': False}},
            {'document_type': None},
            {'document_type': 'unknown'}
        ]
    }))

    print(f"Found {len(questions_to_fix)} questions that need document_type fixes")

    # Group questions by document_id
    by_document_id = {}
    for q in questions_to_fix:
        doc_id = q.get('document_id')
        if doc_id:
            if doc_id not in by_document_id:
                by_document_id[doc_id] = []
            by_document_id[doc_id].append(q)

    print(f"Questions grouped by {len(by_document_id)} document IDs")

    # Get document types for each document_id
    document_types = {}
    for doc_id in by_document_id.keys():
        doc = documents_collection.find_one({'document_id': doc_id})
        if doc:
            document_types[doc_id] = doc.get('document_type', 'Chapter Notes')
        else:
            print(f"Warning: Document {doc_id} not found in documents collection")
            document_types[doc_id] = 'Chapter Notes'  # Default

    # Update questions and ChromaDB
    updated_count = 0
    chromadb_updated_count = 0

    for doc_id, questions in by_document_id.items():
        correct_doc_type = document_types[doc_id]
        print(f"Updating {len(questions)} questions for document {doc_id} to document_type: {correct_doc_type}")

        for question in questions:
            # Update MongoDB
            result = questions_collection.update_one(
                {'id': question['id']},
                {'$set': {'document_type': correct_doc_type}}
            )

            if result.modified_count > 0:
                updated_count += 1

            # Update ChromaDB metadata
            try:
                # Get current ChromaDB entry
                existing = chromadb_client.collection.get(ids=[question['id']])
                if existing['ids']:
                    # Get the current metadata and update document_type
                    current_metadata = existing['metadatas'][0]
                    updated_metadata = current_metadata.copy()
                    updated_metadata['document_type'] = correct_doc_type

                    # Update ChromaDB
                    chromadb_client.collection.update(
                        ids=[question['id']],
                        metadatas=[updated_metadata]
                    )
                    chromadb_updated_count += 1
            except Exception as e:
                print(f"Failed to update ChromaDB for question {question['id']}: {e}")

    print(f"Updated {updated_count} questions in MongoDB")
    print(f"Updated {chromadb_updated_count} questions in ChromaDB")

    # Verify the fixes
    print("\nVerifying fixes...")

    # Check MongoDB
    fixed_questions = list(questions_collection.find({
        'document_type': {'$in': ['Practice Sets', 'Test Series', 'Chapter Notes']}
    }))
    print(f"MongoDB now has {len(fixed_questions)} questions with valid document_types")

    # Check ChromaDB
    try:
        chromadb_results = chromadb_client.collection.get(include=['metadatas'])
        chromadb_by_type = {}
        for metadata in chromadb_results['metadatas']:
            doc_type = metadata.get('document_type', 'unknown')
            if doc_type not in chromadb_by_type:
                chromadb_by_type[doc_type] = 0
            chromadb_by_type[doc_type] += 1

        print("ChromaDB questions by document_type:")
        for doc_type, count in chromadb_by_type.items():
            print(f"  {doc_type}: {count} questions")
    except Exception as e:
        print(f"Failed to verify ChromaDB: {e}")

if __name__ == '__main__':
    fix_question_document_types()
