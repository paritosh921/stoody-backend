# Backend Fixes Applied - Based on Frontend Report

## Summary

All backend fixes recommended by the frontend team have been implemented. The update endpoint now supports all fields that the frontend sends.

---

## Changes Made

### 1. ✅ Added `question_figures` Support

**File:** `api/v1/pdf_async.py:2389-2398`

**What it does:**
- Accepts `question_figures` field in PUT request body
- Validates all question figure image references
- Filters out orphaned/broken images automatically
- Logs warnings for invalid images

**Frontend can now send:**
```javascript
{
  question_figures: [
    {
      id: "image-uuid",
      filename: "image-uuid.png",
      path: "uploads/images/image-uuid.png",
      description: "Question diagram",
      type: "diagram"
    }
  ]
}
```

---

### 2. ✅ Added `enhanced_options` Support

**File:** `api/v1/pdf_async.py:2400-2402`

**What it does:**
- Accepts `enhanced_options` field in PUT request body
- Saves options with full metadata (type, content, label, etc.)
- Supports both text and image-based options

**Frontend can now send:**
```javascript
{
  enhanced_options: [
    {
      id: "opt-a",
      type: "text",
      content: "Option A text",
      label: "A",
      description: "First option"
    },
    {
      id: "opt-b",
      type: "image",
      content: "image-id-here",
      label: "B"
    }
  ]
}
```

---

### 3. ✅ Fixed `document_id` Consistency in Points Recalculation

**File:** `api/v1/pdf_async.py:2465-2476`

**What changed:**
- Now uses `document_id` as primary field (with fallback to `pdf_source` for legacy data)
- Queries questions using `document_id` consistently
- Fallback to `pdf_source` if no questions found with `document_id`
- Ensures total points calculation works regardless of which field is present

**Before:**
```python
document_id = existing_question.get("pdf_source")  # ❌ Inconsistent
all_questions = await db.mongo_find("questions", {"pdf_source": document_id})
```

**After:**
```python
document_id = existing_question.get("document_id") or existing_question.get("pdf_source")  # ✅ Consistent
all_questions = await db.mongo_find("questions", {"document_id": document_id})
if not all_questions:
    all_questions = await db.mongo_find("questions", {"pdf_source": document_id})  # Fallback
```

---

### 4. ✅ ChromaDB Metadata Already Correct

**File:** `api/v1/pdf_async.py:2444-2445`

The ChromaDB metadata update was already correctly including both `images` and `question_figures`:

```python
"hasImages": len(updated_question.get("images", [])) > 0 or len(updated_question.get("question_figures", [])) > 0,
"imageCount": len(updated_question.get("images", [])) + len(updated_question.get("question_figures", [])),
```

No changes needed here.

---

## Complete Update Endpoint Specification

### Endpoint
```
PUT /api/v1/pdf/questions/{question_id}
```

### Supported Fields (All Optional)

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Question text |
| `options` | string[] | Answer options (text) |
| `correct_answer` | string | Correct answer |
| `subject` | string | Subject name |
| `difficulty` | string | "easy" \| "medium" \| "hard" |
| `document_type` | string | Document category |
| `points` | number | Points for question |
| `penalty` | number | Negative marking (max 50) |
| `images` | array | Option images (validated) |
| `question_figures` | array | Question diagrams (validated) ✅ NEW |
| `enhanced_options` | array | Options with metadata ✅ NEW |

---

## Frontend Usage Examples

### Update Question with Diagram Image

```javascript
// 1. Upload image
const formData = new FormData();
formData.append('file', imageFile);

const uploadRes = await fetch('/api/v1/images/upload', {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${token}` },
  body: formData
});

const imageData = await uploadRes.json();

// 2. Update question with question_figures
await fetch(`/api/v1/pdf/questions/${questionId}`, {
  method: 'PUT',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({
    text: "Updated question text",
    question_figures: [{  // ← Use question_figures for diagrams
      id: imageData.id,
      filename: imageData.filename,
      path: `uploads/images/${imageData.filename}`,
      description: "Question diagram",
      type: "diagram"
    }]
  })
});
```

### Update Question with Enhanced Options

```javascript
await fetch(`/api/v1/pdf/questions/${questionId}`, {
  method: 'PUT',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({
    enhanced_options: [
      { id: "opt-a", type: "text", content: "Option A", label: "A" },
      { id: "opt-b", type: "text", content: "Option B", label: "B" },
      { id: "opt-c", type: "image", content: "image-id", label: "C" },
      { id: "opt-d", type: "text", content: "Option D", label: "D" }
    ],
    correct_answer: "B"
  })
});
```

---

## Image Validation

Both `images` and `question_figures` are automatically validated:

✅ **Validated:**
- Image ID exists in database
- Image file exists on filesystem
- Valid image reference format

❌ **Filtered Out:**
- Orphaned images (ID in DB but file missing)
- Invalid image IDs (not in DB)
- Broken references

**Logs warnings** when invalid images are filtered:
```
WARNING: Question 34b0eb73-e57e-463e-8050-784c80045cb2 update attempted with 2 invalid question figures. These will be filtered out: ['img-2', 'img-4']
```

---

## Backward Compatibility

All changes are **100% backward compatible**:

✅ Old requests without new fields still work
✅ Partial updates supported (only send changed fields)
✅ Legacy `pdf_source` field still supported
✅ Existing questions continue to work

---

## Testing

### Test Question Update with All New Fields

```bash
curl -X PUT "http://localhost:8000/api/v1/pdf/questions/34b0eb73-e57e-463e-8050-784c80045cb2" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "text": "Updated question",
    "question_figures": [
      {
        "id": "valid-image-id",
        "filename": "image.png",
        "path": "uploads/images/image.png",
        "description": "Diagram",
        "type": "diagram"
      }
    ],
    "enhanced_options": [
      {"id": "a", "type": "text", "content": "Option A", "label": "A"},
      {"id": "b", "type": "text", "content": "Option B", "label": "B"}
    ]
  }'
```

### Check Backend Logs

```bash
tail -f /home/ubuntu/backend/logs/app.log
```

Expected output:
```
📝 Update question request received for question_id=34b0eb73-e57e-463e-8050-784c80045cb2
   Update data keys: ['text', 'question_figures', 'enhanced_options']
   User: 68d9095ed428b8532fc184ae
```

---

## What Frontend Should Do Now

### Immediate Fix

The frontend code is already sending the correct fields according to the report:
- `images` for option images ✅
- `question_figures` for diagram images ✅
- `enhanced_options` for advanced options ✅

**Backend now accepts and saves all of these!**

### Verify It Works

1. **Clear orphaned images first** (one-time cleanup):
   ```bash
   cd /home/ubuntu/backend
   source .venv/bin/activate
   python clean_orphaned_images.py --document phy006
   ```

2. **Try editing a question** in the frontend:
   - Add a diagram image
   - Save
   - Refresh and verify image shows

3. **Check Network tab** should show:
   ```
   PUT /api/v1/pdf/questions/{id}
   Status: 200 OK
   Response: {"message": "Question updated successfully", "question_id": "..."}
   ```

4. **Check backend logs** should show:
   ```
   📝 Update question request received for question_id=...
      Update data keys: ['text', 'question_figures']
   ```

---

## Files Modified

1. `api/v1/pdf_async.py`
   - Lines 2389-2398: Added `question_figures` validation and save
   - Lines 2400-2402: Added `enhanced_options` save
   - Lines 2465-2476: Fixed `document_id` consistency

2. `FRONTEND_API_GUIDE.md`
   - Updated examples to show `question_figures` usage
   - Updated field reference table
   - Added `enhanced_options` examples

---

## Summary

✅ **All recommended backend fixes implemented**
✅ **No breaking changes - fully backward compatible**
✅ **Image validation ensures data integrity**
✅ **Consistent field naming across endpoints**
✅ **Frontend can now update all question fields**

The backend is ready. Frontend should now be able to save question edits with diagrams successfully!
