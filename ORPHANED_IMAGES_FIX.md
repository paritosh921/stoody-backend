# Orphaned Images Fix

## Problem Summary

Questions in the database were referencing image IDs that don't exist in the backend storage, causing:
- 404 errors when trying to view questions
- Inability to edit/update questions with broken image references
- Poor user experience

## Root Cause

Data inconsistency between:
1. **MongoDB `questions` collection** - Contains image references
2. **MongoDB `images` collection** - Missing image metadata records
3. **Filesystem** - Missing actual image files

This typically happens when:
- Images are deleted from storage but references aren't cleaned up
- Incomplete migrations or backups
- Failed uploads that created metadata but not files

## Solution Implemented

### 1. Image Validation Utility ([utils/image_validator.py](utils/image_validator.py))

Created comprehensive validation functions:
- `validate_image_exists()` - Checks if image exists in DB and filesystem
- `validate_images_list()` - Validates list of image references
- `clean_question_images()` - Removes orphaned references from questions
- `get_orphaned_images_in_question()` - Lists orphaned images in a question
- `get_orphaned_images_in_document()` - Lists orphaned images in a document

### 2. API Endpoints ([api/v1/pdf_async.py](api/v1/pdf_async.py))

Added three new endpoints:

#### GET `/api/v1/pdf/documents/{document_id}/orphaned-images`
Inspect orphaned images without making changes.

**Response:**
```json
{
  "document_id": "phy006",
  "document_title": "Physics Practice",
  "total_orphaned_images": 14,
  "affected_questions": 1,
  "orphaned_by_question": {
    "690603d5edeb5d77f93b20b3": ["img-2", "img-4", "img-5", ...]
  }
}
```

#### POST `/api/v1/pdf/documents/{document_id}/clean-orphaned-images`
Clean all orphaned images from a document.

**Response:**
```json
{
  "message": "Successfully cleaned 14 orphaned image references",
  "document_id": "phy006",
  "questions_cleaned": 1,
  "total_images_removed": 14,
  "details": [...]
}
```

#### POST `/api/v1/pdf/questions/{question_id}/clean-orphaned-images`
Clean orphaned images from a specific question.

**Response:**
```json
{
  "message": "Successfully removed 14 orphaned image references",
  "question_id": "690603d5edeb5d77f93b20b3",
  "removed_count": 14,
  "orphaned_images": ["img-2", "img-4", ...]
}
```

### 3. Auto-Cleaning Features

Modified existing endpoints to automatically handle orphaned images:

#### GET `/api/v1/pdf/documents/{document_id}/questions`
- Automatically cleans orphaned images when retrieving questions
- Updates database silently in the background
- Questions returned are always clean

#### PUT `/api/v1/pdf/questions/{question_id}`
- Validates all image references before saving
- Filters out invalid images automatically
- Logs warning when invalid images are detected

#### GET `/api/v1/pdf/documents/{document_id}/images`
- Filters out orphaned images by default
- Add `?include_orphaned=true` to see all images
- Returns `orphaned_count` in response

### 4. Cleanup Script ([clean_orphaned_images.py](clean_orphaned_images.py))

Command-line tool for immediate cleanup:

```bash
# Scan all documents for orphaned images
python clean_orphaned_images.py --scan

# Clean specific document (your case)
python clean_orphaned_images.py --document phy006

# Clean specific question
python clean_orphaned_images.py --question 690603d5edeb5d77f93b20b3

# Clean all documents
python clean_orphaned_images.py --clean-all
```

## How to Fix Your Issue

### Option 1: Use the Cleanup Script (Recommended)

```bash
cd /home/ubuntu/backend

# If using virtual environment (activate it first)
# source venv/bin/activate  # or your venv path

python3 clean_orphaned_images.py --document phy006
```

This will:
1. Find all orphaned image references in document `phy006`
2. Remove them from the question
3. Update the database
4. Show you a summary of what was cleaned

### Option 2: Use the API Endpoint

```bash
# First, inspect the problem
curl -X GET "http://localhost:8000/api/v1/pdf/documents/phy006/orphaned-images" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Then clean it
curl -X POST "http://localhost:8000/api/v1/pdf/documents/phy006/clean-orphaned-images" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Option 3: Let Auto-Cleaning Handle It

Simply view the document in the frontend - the backend will automatically clean orphaned images when you load the questions.

## Prevention

The following safeguards are now in place:

1. **Automatic validation** - Question updates validate all image references
2. **Auto-cleaning on retrieval** - Questions are cleaned when fetched
3. **Filtered image lists** - Image endpoints skip orphaned images by default
4. **Comprehensive logging** - All cleanup operations are logged

## Technical Details

### Image Validation Logic

1. Check if image exists in `images` collection
2. Check if `file_path` field exists in metadata
3. Normalize the path (handles Windows/Linux differences)
4. Check if file exists on filesystem
5. Return `True` only if all checks pass

### Path Resolution

The validator handles multiple path formats:
- Absolute paths: `/home/ubuntu/backend/uploads/...`
- Relative paths: `uploads/images/...`
- Windows paths: `C:\backend\uploads\...` (converted to POSIX)
- Searches `uploads/` directory tree as fallback

## Files Modified

1. `utils/image_validator.py` - New file with validation utilities
2. `api/v1/pdf_async.py` - Added 3 endpoints + auto-cleaning
3. `clean_orphaned_images.py` - New cleanup script

## Testing

To verify the fix works:

1. Run the scan to see current state:
   ```bash
   python clean_orphaned_images.py --scan
   ```

2. Clean the problematic document:
   ```bash
   python clean_orphaned_images.py --document phy006
   ```

3. Try editing the question in the frontend - it should now work!

## Future Improvements

Consider adding:
1. Periodic cleanup job (cron/celery task)
2. Admin dashboard showing orphaned images
3. Cascade delete when images are removed
4. Image upload retry mechanism
5. Database migration to fix existing data
