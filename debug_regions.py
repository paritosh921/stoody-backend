import asyncio
import os
import sys
from pprint import pprint

sys.path.append(os.getcwd())

from core.database import DatabaseManager

async def check_regions():
    db = DatabaseManager()
    await db.initialize()
    
    # Check mongo database
    print("Checking Mongo DB...")
    doc = await db.mongo_find_one("document_regions", {"document_id": "phy009"})
    if doc:
        print(f"Found regions for phy009: {len(doc.get('regions', []))} regions")
        if doc.get('regions'):
            print("First region sample:")
            pprint(doc['regions'][0])
    else:
        print("No regions found in Mongo for phy009")

    # Check b2c database
    print("\nChecking B2C DB...")
    doc = await db.b2c_find_one("document_regions", {"document_id": "phy009"})
    if doc:
        print(f"Found regions for phy009 (B2C): {len(doc.get('regions', []))} regions")
        if doc.get('regions'):
            print("First region sample:")
            pprint(doc['regions'][0])
    else:
        print("No regions found in B2C for phy009")

if __name__ == "__main__":
    asyncio.run(check_regions())
