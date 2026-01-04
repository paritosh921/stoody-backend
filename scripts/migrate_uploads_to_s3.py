"""
Migration script to move existing local uploads to S3 and update database paths.

This script:
1. Scans the local uploads/ folder for PDFs and images
2. Uploads each file to S3
3. Updates the corresponding MongoDB document with the S3 path

Usage:
    python scripts/migrate_uploads_to_s3.py --dry-run  # Preview what will be migrated
    python scripts/migrate_uploads_to_s3.py            # Actually perform migration
"""

import asyncio
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from utils.s3_storage import (
    is_s3_enabled, 
    upload_file as s3_upload_file, 
    _local_path_to_s3_key,
    S3_BUCKET_NAME
)
from core.database import DatabaseManager


async def migrate_documents_to_s3(db: DatabaseManager, dry_run: bool = True):
    """Migrate documents (PDFs) from local storage to S3"""
    
    backend_dir = Path(__file__).parent.parent
    uploads_dir = backend_dir / "uploads"
    
    print(f"\n{'='*60}")
    print("MIGRATING DOCUMENTS (PDFs) TO S3")
    print(f"{'='*60}")
    
    # Get all documents from MongoDB
    documents = await db.mongo_find("documents", {})
    b2c_documents = await db.b2c_find("documents", {})
    
    all_docs = [("main", doc) for doc in documents] + [("b2c", doc) for doc in b2c_documents]
    
    migrated = 0
    skipped = 0
    failed = 0
    already_s3 = 0
    
    for db_type, doc in all_docs:
        doc_id = doc.get("document_id") or str(doc.get("_id"))
        file_path = doc.get("file_path", "")
        
        # Skip if already on S3
        if file_path.startswith("s3://"):
            already_s3 += 1
            continue
        
        # Resolve local path
        if file_path.startswith("uploads/"):
            local_path = backend_dir / file_path.replace("/", os.sep)
        else:
            local_path = Path(file_path)
        
        if not local_path.exists():
            print(f"  ⚠️  File not found: {local_path} (doc: {doc_id})")
            skipped += 1
            continue
        
        s3_key = _local_path_to_s3_key(str(local_path))
        s3_path = f"s3://{S3_BUCKET_NAME}/{s3_key}"
        
        if dry_run:
            print(f"  [DRY RUN] Would migrate: {doc_id}")
            print(f"             From: {local_path}")
            print(f"             To:   {s3_path}")
            migrated += 1
        else:
            try:
                # Read file
                with open(local_path, "rb") as f:
                    file_data = f.read()
                
                # Upload to S3
                success, storage_path = await s3_upload_file(
                    file_data=file_data,
                    local_path=str(local_path),
                    content_type="application/pdf"
                )
                
                if success:
                    # Update database with S3 path
                    query = {"_id": doc.get("_id")}
                    update = {"$set": {"file_path": storage_path, "is_s3": True}}
                    
                    if db_type == "main":
                        await db.mongo_update_one("documents", query, update)
                    else:
                        await db.b2c_update_one("documents", query, update)
                    
                    print(f"  ✅ Migrated: {doc_id} -> {s3_key}")
                    migrated += 1
                else:
                    print(f"  ❌ S3 upload failed: {doc_id}")
                    failed += 1
                    
            except Exception as e:
                print(f"  ❌ Error migrating {doc_id}: {e}")
                failed += 1
    
    print(f"\nDocuments Summary:")
    print(f"  Already on S3: {already_s3}")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    
    return migrated, skipped, failed


async def migrate_images_to_s3(db: DatabaseManager, dry_run: bool = True):
    """Migrate images from local storage to S3"""
    
    backend_dir = Path(__file__).parent.parent
    
    print(f"\n{'='*60}")
    print("MIGRATING IMAGES TO S3")
    print(f"{'='*60}")
    
    # Get all images from MongoDB
    images = await db.mongo_find("images", {})
    b2c_images = await db.b2c_find("images", {})
    
    all_images = [("main", img) for img in images] + [("b2c", img) for img in b2c_images]
    
    migrated = 0
    skipped = 0
    failed = 0
    already_s3 = 0
    
    for db_type, img in all_images:
        img_id = img.get("_id")
        file_path = img.get("file_path", "")
        
        # Skip if already on S3
        if file_path.startswith("s3://"):
            already_s3 += 1
            continue
        
        # Resolve local path
        if file_path.startswith("uploads/"):
            local_path = backend_dir / file_path.replace("/", os.sep)
        elif file_path.startswith("/"):
            local_path = Path(file_path)
        else:
            local_path = backend_dir / file_path
        
        if not local_path.exists():
            # Try finding by filename in pdf_images
            filename = img.get("filename", "")
            source_pdf = img.get("source_pdf", "")
            if source_pdf and filename:
                pdf_name = source_pdf.replace(".pdf", "")
                alt_path = backend_dir / "uploads" / "pdf_images" / pdf_name / filename
                if alt_path.exists():
                    local_path = alt_path
        
        if not local_path.exists():
            skipped += 1
            continue
        
        s3_key = _local_path_to_s3_key(str(local_path))
        s3_path = f"s3://{S3_BUCKET_NAME}/{s3_key}"
        
        # Detect content type
        ext = local_path.suffix.lower()
        content_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", 
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        content_type = content_type_map.get(ext, "image/jpeg")
        
        if dry_run:
            migrated += 1
        else:
            try:
                with open(local_path, "rb") as f:
                    file_data = f.read()
                
                success, storage_path = await s3_upload_file(
                    file_data=file_data,
                    local_path=str(local_path),
                    content_type=content_type
                )
                
                if success:
                    query = {"_id": img_id}
                    update = {"$set": {"file_path": storage_path, "is_s3": True}}
                    
                    if db_type == "main":
                        await db.mongo_update_one("images", query, update)
                    else:
                        await db.b2c_update_one("images", query, update)
                    
                    migrated += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"  ❌ Error migrating image {img_id}: {e}")
                failed += 1
    
    print(f"\nImages Summary:")
    print(f"  Already on S3: {already_s3}")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped (file not found): {skipped}")
    print(f"  Failed: {failed}")
    
    return migrated, skipped, failed


async def main():
    parser = argparse.ArgumentParser(description="Migrate local uploads to S3")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without migrating")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("S3 MIGRATION TOOL")
    print(f"{'='*60}")
    print(f"Mode: {'DRY RUN (no changes)' if args.dry_run else 'LIVE MIGRATION'}")
    print(f"Time: {datetime.utcnow().isoformat()}")
    
    # Check S3 is enabled
    if not is_s3_enabled():
        print("\n❌ S3 is not enabled! Check your environment variables:")
        print("   - USE_S3_STORAGE=true")
        print("   - AWS_ACCESS_KEY_ID")
        print("   - AWS_SECRET_ACCESS_KEY")
        print("   - S3_BUCKET_NAME")
        sys.exit(1)
    
    print(f"\n✅ S3 Enabled - Bucket: {S3_BUCKET_NAME}")
    
    # Initialize database
    db = DatabaseManager()
    await db.connect()
    
    try:
        # Migrate documents
        doc_migrated, doc_skipped, doc_failed = await migrate_documents_to_s3(db, args.dry_run)
        
        # Migrate images
        img_migrated, img_skipped, img_failed = await migrate_images_to_s3(db, args.dry_run)
        
        print(f"\n{'='*60}")
        print("MIGRATION COMPLETE")
        print(f"{'='*60}")
        print(f"Documents migrated: {doc_migrated}")
        print(f"Images migrated: {img_migrated}")
        print(f"Total migrated: {doc_migrated + img_migrated}")
        
        if not args.dry_run:
            print("\n🎉 Migration completed successfully!")
            print("   New uploads will automatically go to S3.")
            print("   You can safely remove the local uploads/ folder after verification.")
        else:
            print("\n📋 This was a dry run. No changes were made.")
            print("   Run without --dry-run to perform the actual migration.")
            
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
