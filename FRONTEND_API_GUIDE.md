# Frontend API Integration Guide - Question Update

## Problem: Question Updates Not Saving to Backend

The frontend is currently calling `GET` on the question endpoint instead of `PUT`, which only reads the question but doesn't save changes.

---

## Complete Flow to Update Existing Questions

### Step 1: Upload New Images (if adding/changing images)

**Endpoint:** `POST /api/v1/images/upload`

**Purpose:** Upload image files first, get image IDs to reference in question

**Request:**
```javascript
const formData = new FormData();
formData.append('file', imageFile); // File object from <input type="file">

const response = await fetch('/api/v1/images/upload', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${authToken}` // Required
  },
  body: formData // Don't set Content-Type header, browser sets it automatically
});

const result = await response.json();
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "550e8400-e29b-41d4-a716-446655440000.png",
  "url": "/api/v1/images/550e8400-e29b-41d4-a716-446655440000",
  "size": 45678,
  "content_type": "image/png",
  "uploaded_at": "2025-11-08T12:00:00.000000"
}
```

**Save the `id` field** - you'll use it in Step 2.

**Allowed file types:**
- Images: `.jpg`, `.jpeg`, `.png`, `.gif`
- Documents: `.pdf`, `.doc`, `.docx`, `.txt`

**Max file size:** 10MB

---

### Step 2: Update the Question

**Endpoint:** `PUT /api/v1/pdf/questions/{question_id}`

**Purpose:** Update question text, options, images, and other fields

**Request:**
```javascript
const questionId = "34b0eb73-e57e-463e-8050-784c80045cb2"; // From your network tab

const updateData = {
  // Question text (optional - only include if changed)
  text: "Updated question text here",

  // Options (optional - only include if changed)
  options: ["Option A", "Option B", "Option C", "Option D"],

  // Correct answer (optional)
  correct_answer: "Option B",

  // Subject (optional)
  subject: "Physics",

  // Difficulty (optional)
  difficulty: "medium", // "easy" | "medium" | "hard"

  // Document type (optional)
  document_type: "Practice Sets",

  // Points (optional)
  points: 1.0,

  // Penalty (optional, max 50)
  penalty: 0,

  // Images for options - IMPORTANT: Must be in this format
  images: [
    {
      id: "550e8400-e29b-41d4-a716-446655440000", // From Step 1
      filename: "550e8400-e29b-41d4-a716-446655440000.png",
      path: "uploads/images/550e8400-e29b-41d4-a716-446655440000.png",
      description: "Option image",
      type: "option"
    }
  ],

  // Question diagram/figure images - separate from option images
  question_figures: [
    {
      id: "another-uuid-from-upload", // From Step 1
      filename: "another-uuid.png",
      path: "uploads/images/another-uuid.png",
      description: "Question diagram",
      type: "diagram"
    }
  ],

  // Enhanced options (with metadata/images)
  enhanced_options: [
    {
      id: "opt-a",
      type: "text",
      content: "Option A text",
      label: "A"
    },
    {
      id: "opt-b",
      type: "image",
      content: "image-id-here",
      label: "B"
    }
  ]
};

const response = await fetch(`/api/v1/pdf/questions/${questionId}`, {
  method: 'PUT', // ← MUST be PUT, not GET!
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${authToken}` // Required
  },
  body: JSON.stringify(updateData)
});

const result = await response.json();
```

**Response (Success):**
```json
{
  "message": "Question updated successfully",
  "question_id": "34b0eb73-e57e-463e-8050-784c80045cb2"
}
```

**Response (Error):**
```json
{
  "detail": "Error message here"
}
```

---

## Complete Example: Update Question with New Image

```javascript
async function updateQuestionWithImage(questionId, newImage, newQuestionText) {
  try {
    // Step 1: Upload the image
    const formData = new FormData();
    formData.append('file', newImage);

    const uploadResponse = await fetch('/api/v1/images/upload', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('authToken')}`
      },
      body: formData
    });

    if (!uploadResponse.ok) {
      const error = await uploadResponse.json();
      throw new Error(`Image upload failed: ${error.detail}`);
    }

    const imageData = await uploadResponse.json();
    console.log('Image uploaded:', imageData);

    // Step 2: Update the question
    const updateData = {
      text: newQuestionText,
      question_figures: [  // Use question_figures for diagram images
        {
          id: imageData.id,
          filename: imageData.filename,
          path: `uploads/images/${imageData.filename}`,
          description: "Question diagram",
          type: "diagram"
        }
      ]
    };

    const updateResponse = await fetch(`/api/v1/pdf/questions/${questionId}`, {
      method: 'PUT', // ← IMPORTANT: Use PUT, not GET!
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('authToken')}`
      },
      body: JSON.stringify(updateData)
    });

    if (!updateResponse.ok) {
      const error = await updateResponse.json();
      throw new Error(`Question update failed: ${error.detail}`);
    }

    const result = await updateResponse.json();
    console.log('Question updated successfully:', result);

    return result;

  } catch (error) {
    console.error('Error updating question:', error);
    throw error;
  }
}

// Usage:
const questionId = "34b0eb73-e57e-463e-8050-784c80045cb2";
const imageFile = document.getElementById('imageInput').files[0];
const newText = "Updated question text";

updateQuestionWithImage(questionId, imageFile, newText)
  .then(() => {
    alert('Question saved successfully!');
  })
  .catch(error => {
    alert(`Error: ${error.message}`);
  });
```

---

## Update Question Text Only (No Images)

If you just want to update the question text without adding images:

```javascript
async function updateQuestionText(questionId, newText) {
  const response = await fetch(`/api/v1/pdf/questions/${questionId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${localStorage.getItem('authToken')}`
    },
    body: JSON.stringify({
      text: newText
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }

  return await response.json();
}
```

---

## Update Options Only

```javascript
async function updateQuestionOptions(questionId, newOptions, correctAnswer) {
  const response = await fetch(`/api/v1/pdf/questions/${questionId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${localStorage.getItem('authToken')}`
    },
    body: JSON.stringify({
      options: newOptions,
      correct_answer: correctAnswer
    })
  });

  return await response.json();
}

// Usage:
updateQuestionOptions(
  "34b0eb73-e57e-463e-8050-784c80045cb2",
  ["New Option A", "New Option B", "New Option C", "New Option D"],
  "New Option B"
);
```

---

## Important Notes

### 1. Authentication Required
All requests MUST include the `Authorization` header with a valid Bearer token:
```javascript
headers: {
  'Authorization': `Bearer ${authToken}`
}
```

### 2. Only Send Changed Fields
The backend accepts partial updates. You don't need to send all fields, only the ones you want to change:
```javascript
// ✅ Good - only updating text
{ text: "New text" }

// ❌ Wasteful - sending everything even if unchanged
{ text: "New text", options: [...], subject: "Physics", ... }
```

### 3. Image References Format
Images must be in this exact format:
```javascript
{
  id: "uuid-from-upload",           // Required - from upload response
  filename: "uuid.png",               // Required - from upload response
  path: "uploads/images/uuid.png",    // Required - construct from filename
  description: "Description",         // Optional
  type: "diagram"                     // Optional - "diagram", "graph", "photo", etc.
}
```

### 4. Backend Auto-Validates Images
The backend automatically:
- ✅ Validates image IDs exist
- ✅ Checks files exist on disk
- ✅ Filters out orphaned/broken images
- ✅ Logs warnings for invalid images

You'll never accidentally save broken image references!

---

## Common Errors and Solutions

### Error: 401 Unauthorized
**Problem:** Missing or invalid auth token
**Solution:** Check `Authorization` header is set correctly

### Error: 404 Not Found (Question)
**Problem:** Question ID doesn't exist
**Solution:** Verify the question ID is correct

### Error: 404 Not Found (Image)
**Problem:** Image ID doesn't exist in database
**Solution:** Upload the image first (Step 1) before updating question

### Error: 400 Bad Request
**Problem:** Invalid request data
**Solution:** Check request body format matches examples above

### Error: 403 Forbidden
**Problem:** User doesn't have admin permissions
**Solution:** Login as admin user

---

## Question Fields Reference

All fields you can update:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `text` | string | Question text | "Q. 6 Two long parallel wires..." |
| `options` | array | Answer options (text) | ["Option A", "Option B", ...] |
| `correct_answer` | string | Correct answer | "Option B" |
| `subject` | string | Subject name | "Physics" |
| `difficulty` | string | Difficulty level | "easy", "medium", "hard" |
| `document_type` | string | Document category | "Practice Sets", "Test Series", etc. |
| `images` | array | Option image references | See format above |
| `question_figures` | array | Question diagram images | See format above |
| `enhanced_options` | array | Options with metadata/images | See format above |
| `points` | number | Points for question | 1.0 |
| `penalty` | number | Negative marking (max 50) | 0 |

**You only need to include the fields you want to update!**

---

## Testing Your Implementation

### 1. Test Image Upload
```bash
curl -X POST "http://localhost:8000/api/v1/images/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@test-image.png"
```

### 2. Test Question Update
```bash
curl -X PUT "http://localhost:8000/api/v1/pdf/questions/34b0eb73-e57e-463e-8050-784c80045cb2" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"text": "Updated question text"}'
```

### 3. Check Backend Logs
After making requests, check:
```bash
tail -f /home/ubuntu/backend/logs/app.log
```

You should see:
```
📝 Update question request received for question_id=...
   Update data keys: ['text', 'images']
   User: 68d9095ed428b8532fc184ae
```

---

## Quick Fix Checklist

Replace this in your frontend code:

❌ **WRONG - Current Code:**
```javascript
// This only READS the question, doesn't save!
fetch(`/api/v1/pdf/questions/${questionId}`, {
  method: 'GET'
})
```

✅ **CORRECT - Fixed Code:**
```javascript
// This actually SAVES the question
fetch(`/api/v1/pdf/questions/${questionId}`, {
  method: 'PUT',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${authToken}`
  },
  body: JSON.stringify(updateData)
})
```

---

## Need Help?

1. **Check browser console** for JavaScript errors
2. **Check Network tab** to see actual requests being sent
3. **Check backend logs** at `/home/ubuntu/backend/logs/app.log`
4. **Test with curl** to verify backend works independently

The backend is **fully functional and ready**. The frontend just needs to call the correct endpoint with the correct HTTP method!
