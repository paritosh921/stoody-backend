# QUICK FIX: Frontend Question Update Issue

## The Problem

Frontend is calling `GET /api/v1/pdf/questions/{id}` instead of `PUT`, so changes are only saved to localStorage, not the database.

---

## The Solution

### Change 1 Line of Code:

**Before (WRONG):**
```javascript
method: 'GET'  // ❌ This only reads, doesn't save
```

**After (CORRECT):**
```javascript
method: 'PUT'  // ✅ This actually saves
```

---

## Complete Working Example

```javascript
// When user clicks "Save" button after editing question
async function saveQuestionChanges(questionId, updatedData) {
  const response = await fetch(`/api/v1/pdf/questions/${questionId}`, {
    method: 'PUT',  // ← Changed from GET to PUT
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${localStorage.getItem('authToken')}`
    },
    body: JSON.stringify(updatedData)
  });

  if (!response.ok) {
    throw new Error('Failed to save question');
  }

  return await response.json();
}

// Example usage:
saveQuestionChanges("34b0eb73-e57e-463e-8050-784c80045cb2", {
  text: "Updated question text",
  options: ["A", "B", "C", "D"],
  correct_answer: "B"
})
.then(() => alert('Saved!'))
.catch(err => alert('Error: ' + err.message));
```

---

## If Adding Images

### Step 1: Upload Image First
```javascript
const formData = new FormData();
formData.append('file', imageFile);

const uploadRes = await fetch('/api/v1/images/upload', {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${token}` },
  body: formData
});

const imageData = await uploadRes.json();
// Save imageData.id for next step
```

### Step 2: Update Question with Image Reference
```javascript
await fetch(`/api/v1/pdf/questions/${questionId}`, {
  method: 'PUT',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({
    text: "Question text",
    images: [{
      id: imageData.id,  // From Step 1
      filename: imageData.filename,
      path: `uploads/images/${imageData.filename}`,
      description: "Question image",
      type: "diagram"
    }]
  })
});
```

---

## Test It Works

### 1. Open Browser DevTools → Network Tab
### 2. Edit a question and click Save
### 3. Look for this request:

✅ **Should see:**
```
PUT /api/v1/pdf/questions/34b0eb73-e57e-463e-8050-784c80045cb2
Status: 200 OK
```

❌ **Don't want to see:**
```
GET /api/v1/pdf/questions/34b0eb73-e57e-463e-8050-784c80045cb2
Status: 200 OK
```

---

## Verify Backend Received It

Check logs:
```bash
tail -f /home/ubuntu/backend/logs/app.log
```

Should see:
```
📝 Update question request received for question_id=34b0eb73-e57e-463e-8050-784c80045cb2
   Update data keys: ['text', 'options']
   User: 68d9095ed428b8532fc184ae
```

---

## Summary

**Current Issue:** Frontend uses `GET` method = only reads question
**Solution:** Change to `PUT` method = actually saves question

**Backend is working perfectly.** Just need to fix the frontend HTTP method.

See [FRONTEND_API_GUIDE.md](FRONTEND_API_GUIDE.md) for complete documentation.
