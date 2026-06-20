# S3 Storage Migration Guide

This guide explains how to migrate from local filesystem storage to Amazon S3 for PDFs and extracted images.

## 📁 Current Storage Structure

Currently, files are stored locally on the EC2 instance:

```
s-backend/
  uploads/
    documents/
      Chapter Notes/     # PDF files
      Practice Sets/     # PDF files  
      Test Series/       # PDF files
    pdf_images/
      doc001/            # Extracted question images
      doc002/
        img-1.jpeg
        img-2.jpeg
```

## 🚀 S3 Storage Benefits

1. **Scalability** - Unlimited storage vs EC2 disk limits
2. **Durability** - 99.999999999% (11 nines) durability
3. **Availability** - Files survive EC2 restarts/replacements
4. **Cost** - Pay only for what you use (~$0.023/GB/month)
5. **Performance** - Use CloudFront CDN for faster delivery
6. **Backup** - S3 versioning for accidental deletion protection

## ⚙️ Setup Instructions

### Step 1: Create S3 Bucket

```bash
# Via AWS CLI
aws s3 mb s3://stoody-assets-prod --region ap-south-1

# Enable versioning (recommended)
aws s3api put-bucket-versioning \
    --bucket stoody-assets-prod \
    --versioning-configuration Status=Enabled
```

### Step 2: Configure CORS (for browser uploads)

Create `cors.json`:
```json
{
    "CORSRules": [
        {
            "AllowedHeaders": ["*"],
            "AllowedMethods": ["GET", "PUT", "POST"],
            "AllowedOrigins": ["https://yourdomain.com", "http://localhost:8080"],
            "ExposeHeaders": ["ETag"]
        }
    ]
}
```

Apply CORS:
```bash
aws s3api put-bucket-cors --bucket stoody-assets-prod --cors-configuration file://cors.json
```

### Step 3: Create IAM User/Role

Create an IAM policy:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject",
                "s3:HeadObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::stoody-assets-prod",
                "arn:aws:s3:::stoody-assets-prod/*"
            ]
        }
    ]
}
```

### Step 4: Add Environment Variables

Add to your `.env` file or EC2 environment:

```bash
# S3 Configuration
USE_S3_STORAGE=true
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=ap-south-1
S3_BUCKET_NAME=stoody-assets-prod

# Optional: CloudFront for faster delivery
CLOUDFRONT_DOMAIN=d1234567890.cloudfront.net
```

### Step 5: Install boto3

```bash
pip install boto3
# Or add to requirements.txt:
# boto3>=1.34.0
```

### Step 6: Update Code (Minimal Changes)

The `s3_storage.py` module provides drop-in functions. Update these files:

#### In `pdf_async.py` (save_image_to_disk function):

```python
# Replace local file writes with:
from utils.s3_storage import upload_file

# OLD:
async with aiofiles.open(original_file_path, "wb") as f:
    await f.write(image_data)

# NEW:
success, storage_path = await upload_file(
    file_data=image_data,
    local_path=original_file_path,
    content_type=original_content_type
)
```

#### In MongoDB, store the `storage_path` instead of local path:

```python
original_metadata = {
    "_id": base_image_id,
    "filename": original_image_filename,
    "file_path": storage_path,  # Now contains s3://... or local path
    # ... rest of metadata
}
```

#### When serving files:

```python
from utils.s3_storage import get_public_url

# Get URL to serve to client
url = get_public_url(document["file_path"])
```

## 🔄 Migration Script

Run this to migrate existing files to S3:

```python
# migrate_to_s3.py
import asyncio
from utils.s3_storage import migrate_directory_to_s3

async def main():
    # First, do a dry run
    print("=== DRY RUN ===")
    report = await migrate_directory_to_s3("uploads", dry_run=True)
    print(f"Would migrate {report['total_files']} files")
    
    # Then actually migrate
    confirm = input("Proceed with migration? (yes/no): ")
    if confirm.lower() == "yes":
        print("=== MIGRATING ===")
        report = await migrate_directory_to_s3("uploads", dry_run=False)
        print(f"Migrated: {report['migrated']}")
        print(f"Failed: {report['failed']}")
        if report['errors']:
            print("Errors:", report['errors'])

asyncio.run(main())
```

## 📊 S3 Bucket Structure

Files will be organized in S3 as:

```
stoody-assets-prod/
  documents/
    Chapter Notes/
      doc001.pdf
    Practice Sets/
      practice001.pdf
    Test Series/
      test001.pdf
  pdf_images/
    doc001/
      img-1.jpeg
      img-2.jpeg
```

## ⚡ Optional: CloudFront CDN

For faster global delivery:

1. Create CloudFront Distribution
2. Set S3 bucket as origin
3. Enable origin access identity (OAI)
4. Add `CLOUDFRONT_DOMAIN` to environment

## 🔒 Security Best Practices

1. **Never commit AWS credentials** - Use environment variables
2. **Use IAM roles** for EC2 instead of access keys when possible
3. **Enable bucket versioning** for accidental deletion protection
4. **Enable server-side encryption** (SSE-S3 or SSE-KMS)
5. **Keep bucket private** - Use presigned URLs for access

## Upload Security Note

Generated derivatives can still use the legacy S3/local abstraction where the
calling route proves they are not raw user uploads. Raw user-controlled uploads
must go through `core/upload_security.secure_upload()` first and release clean
objects to private storage.

Production local fallback to public `backend/uploads` is disabled by default.
Only set `UPLOAD_ALLOW_PUBLIC_LOCAL_FALLBACK=true` for an explicit development
or emergency procedure, and do not serve raw uploads through `/uploads` in
production.

## ✅ Testing Checklist

- [ ] S3 bucket created and configured
- [ ] IAM credentials work (`aws s3 ls s3://your-bucket`)
- [ ] Environment variables set
- [ ] boto3 installed
- [ ] Dry-run migration works
- [ ] File uploads go to S3
- [ ] File downloads work from S3
- [ ] Images display correctly in frontend
- [ ] PDFs download correctly
- [ ] Old local files backed up before deletion

## 🔄 Rollback Plan

If issues occur:

1. Set `USE_S3_STORAGE=false` to revert to local storage
2. Keep local `uploads/` folder as a temporary backup until migration is verified, but do not use it as a production fallback for new raw user uploads.
   Production S3 upload failures fail closed unless `UPLOAD_ALLOW_PUBLIC_LOCAL_FALLBACK=true` is explicitly set.
3. The storage module will automatically fall back to local if S3 fails
