#!/usr/bin/env python3
"""
Check documents in the database
"""

from models import get_collection

def check_documents():
    docs = list(get_collection('documents').find({}, {
        'document_id':1,
        'title':1,
        'document_type':1,
        'extracted_questions_count':1,
        'ocr_status':1
    }))

    print('Documents in database:')
    for doc in docs:
        print(f"  {doc['document_id']}: {doc['title']} (type: {doc.get('document_type', 'unknown')}, questions: {doc.get('extracted_questions_count', 0)}, status: {doc.get('ocr_status', 'unknown')})")

if __name__ == '__main__':
    check_documents()