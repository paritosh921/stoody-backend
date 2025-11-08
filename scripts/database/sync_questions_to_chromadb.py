#!/usr/bin/env python3
"""
Sync questions from MongoDB to ChromaDB
"""

from models import get_collection, get_chromadb_client
from datetime import datetime

def sync_questions_to_chromadb():
    """Add questions from MongoDB to ChromaDB if they don't exist"""

    questions_collection = get_collection('questions')
    chromadb_client = get_chromadb_client()

    # Get all questions from MongoDB
    mongodb_questions = list(questions_collection.find({}))
    print(f"Found {len(mongodb_questions)} questions in MongoDB")

    # Get all questions from ChromaDB
    chromadb_results = chromadb_client.collection.get(include=['metadatas'])
    chromadb_ids = set(chromadb_results['ids'])
    print(f"Found {len(chromadb_ids)} questions in ChromaDB")

    # Find questions that exist in MongoDB but not in ChromaDB
    missing_in_chromadb = []
    for q in mongodb_questions:
        if q['id'] not in chromadb_ids:
            missing_in_chromadb.append(q)

    print(f"Found {len(missing_in_chromadb)} questions missing from ChromaDB")

    if not missing_in_chromadb:
        print("All questions are already synced!")
        return

    # Add missing questions to ChromaDB
    ids_to_add = []
    documents_to_add = []
    metadatas_to_add = []

    for question in missing_in_chromadb:
        question_id = question['id']
        question_text = question.get('text', '')
        document_id = question.get('document_id', question_id)
        document_type = question.get('document_type', 'Chapter Notes')
        subject = question.get('subject', 'General')
        difficulty = question.get('difficulty', 'medium')

        # Create ChromaDB metadata
        chromadb_metadata = {
            "document_id": document_id,
            "document_type": document_type,
            "subject": subject,
            "difficulty": difficulty,
            "hasImages": len(question.get('images', [])) > 0 or len(question.get('question_figures', [])) > 0,
            "imageCount": len(question.get('images', [])) + len(question.get('question_figures', [])),
            "source": "ocr_sync",
            "created_at": datetime.utcnow().isoformat()
        }

        ids_to_add.append(question_id)
        documents_to_add.append(question_text)
        metadatas_to_add.append(chromadb_metadata)

    print(f"Adding {len(ids_to_add)} questions to ChromaDB...")

    # Add in batches to avoid issues
    batch_size = 10
    added_count = 0

    for i in range(0, len(ids_to_add), batch_size):
        batch_ids = ids_to_add[i:i+batch_size]
        batch_documents = documents_to_add[i:i+batch_size]
        batch_metadatas = metadatas_to_add[i:i+batch_size]

        try:
            chromadb_client.collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            added_count += len(batch_ids)
            print(f"Added batch {i//batch_size + 1}: {len(batch_ids)} questions")
        except Exception as e:
            print(f"Failed to add batch {i//batch_size + 1}: {e}")

    print(f"Successfully added {added_count} questions to ChromaDB")

    # Verify the sync
    print("\nVerifying sync...")
    chromadb_results_after = chromadb_client.collection.get(include=['metadatas'])
    chromadb_by_type = {}
    for metadata in chromadb_results_after['metadatas']:
        doc_type = metadata.get('document_type', 'unknown')
        if doc_type not in chromadb_by_type:
            chromadb_by_type[doc_type] = 0
        chromadb_by_type[doc_type] += 1

    print("ChromaDB questions by document_type after sync:")
    for doc_type, count in chromadb_by_type.items():
        print(f"  {doc_type}: {count} questions")

if __name__ == '__main__':
    sync_questions_to_chromadb()
