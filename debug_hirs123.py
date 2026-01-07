import asyncio
import os
import sys
from pprint import pprint

sys.path.append(os.getcwd())

from core.database import DatabaseManager

async def check_document():
    db = DatabaseManager()
    await db.initialize()
    
    doc_id = "hirs123" 
    
    # Check mongo
    print(f"Checking for {doc_id} in Mongo...")
    doc = await db.mongo_find_one("documents", {"document_id": doc_id})
    if doc:
        print(f"Found document: {doc.get('title', 'No Title')}")
        regions_doc = await db.mongo_find_one("document_regions", {"document_id": doc_id})
        if regions_doc:
            print(f"Found {len(regions_doc.get('regions', []))} regions")
            if regions_doc.get('regions'):
                print("Sample region:")
                pprint(regions_doc['regions'][0])
        else:
            print("No regions document found")
    else:
        print("Document not found in Mongo")

    # Check B2C
    print(f"\nChecking for {doc_id} in B2C...")
    doc = await db.b2c_find_one("documents", {"document_id": doc_id})
    if doc:
        print(f"Found document: {doc.get('title', 'No Title')}")
        regions_doc = await db.b2c_find_one("document_regions", {"document_id": doc_id})
        if regions_doc:
            print(f"Found {len(regions_doc.get('regions', []))} regions")
        else:
            print("No regions document found")
    else:
        print("Document not found in B2C")

if __name__ == "__main__":
    asyncio.run(check_document())
