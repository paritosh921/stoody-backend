# SkillBot Backend API - Complete Endpoint Documentation

## Overview

SkillBot is an educational platform backend built with Flask that provides comprehensive learning tools including question practice, AI-powered chat, multiple-choice questions, and student performance analytics.

**Base URL:** `http://localhost:5001` (configurable via `FLASK_PORT`)

**API Version:** 2.0.0

**Authentication:** JWT-based with role-based access control (Admin/Student)

---

## Table of Contents

1. [Authentication Endpoints](#authentication-endpoints)
2. [Admin Management Endpoints](#admin-management-endpoints)
3. [Student Endpoints](#student-endpoints)
4. [Question Management Endpoints](#question-management-endpoints)
5. [Image Management Endpoints](#image-management-endpoints)
6. [Chat Endpoints](#chat-endpoints)
7. [Practice Mode Endpoints](#practice-mode-endpoints)
8. [MCQ Mode Endpoints](#mcq-mode-endpoints)
9. [Debugger Mode Endpoints](#debugger-mode-endpoints)
10. [Health Check Endpoints](#health-check-endpoints)
11. [Data Models](#data-models)
12. [Error Handling](#error-handling)

---

## Authentication Endpoints

### POST /api/auth/admin/login
**Admin login with email and password**

**Request Body:**
```json
{
  "email": "admin@skillbot.app",
  "password": "admin123"
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Login successful",
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "user_type": "admin",
    "user": {
      "id": "507f1f77bcf86cd799439011",
      "email": "admin@skillbot.app",
      "name": "Admin User",
      "role": "super_admin"
    }
  }
}
```

**Authentication:** None required

---

### POST /api/auth/student/login
**Student login with username and password**

**Request Body:**
```json
{
  "username": "student001",
  "password": "password123"
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Login successful",
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "user_type": "student",
    "user": {
      "id": "507f1f77bcf86cd799439011",
      "student_id": "STU001",
      "username": "student001",
      "name": "John Doe",
      "email": "john@example.com",
      "grade": "10th",
      "school": "Example School"
    }
  }
}
```

**Authentication:** None required

---

### GET /api/auth/verify
**Verify JWT token and return user information**

**Response (Success):**
```json
{
  "success": true,
  "data": {
    "user_type": "student",
    "user": {
      "id": "507f1f77bcf86cd799439011",
      "student_id": "STU001",
      "username": "student001",
      "name": "John Doe",
      "email": "john@example.com",
      "grade": "10th",
      "school": "Example School"
    }
  }
}
```

**Authentication:** JWT token required

---

### POST /api/auth/logout
**Logout user (client-side token invalidation)**

**Response (Success):**
```json
{
  "success": true,
  "message": "Logout successful"
}
```

**Authentication:** JWT token required

---

### POST /api/auth/init-admin
**Initialize default admin account (development only)**

**Response (Success):**
```json
{
  "success": true,
  "message": "Default admin account initialized",
  "data": {
    "email": "admin@skillbot.app",
    "message": "Use email: admin@skillbot.app and password: admin123 to login"
  }
}
```

**Authentication:** None required (development only)

---

## Admin Management Endpoints

### GET /api/admin/students
**Get all students with pagination and search**

**Query Parameters:**
- `page` (int, default: 1): Page number
- `limit` (int, default: 50, max: 100): Items per page
- `search` (string, optional): Search term for student names/emails

**Response (Success):**
```json
{
  "success": true,
  "message": "Students retrieved successfully",
  "data": {
    "students": [
      {
        "student_id": "STU001",
        "name": "John Doe",
        "email": "john@example.com",
        "username": "student001",
        "grade": "10th",
        "school": "Example School",
        "is_active": true,
        "created_at": "2024-01-15T10:30:00Z",
        "last_login": "2024-01-20T14:25:00Z"
      }
    ],
    "total": 150,
    "page": 1,
    "limit": 50,
    "pages": 3
  }
}
```

**Authentication:** Admin JWT required

---

### POST /api/admin/students
**Create a new student**

**Request Body:**
```json
{
  "student_id": "STU002",
  "username": "student002",
  "password": "securepass123",
  "name": "Jane Smith",
  "email": "jane@example.com",
  "grade": "9th",
  "school": "Example School",
  "phone": "+1234567890"
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Student created successfully",
  "data": {
    "student_id": "STU002",
    "student": {
      "student_id": "STU002",
      "username": "student002",
      "name": "Jane Smith",
      "email": "jane@example.com",
      "grade": "9th",
      "school": "Example School",
      "is_active": true,
      "created_at": "2024-01-20T15:30:00Z"
    }
  }
}
```

**Authentication:** Admin JWT required

---

### GET /api/admin/students/{student_id}
**Get student details by ID**

**Path Parameters:**
- `student_id`: Student ID or MongoDB ObjectId

**Response (Success):**
```json
{
  "success": true,
  "message": "Student retrieved successfully",
  "data": {
    "student": {
      "student_id": "STU001",
      "name": "John Doe",
      "email": "john@example.com",
      "username": "student001",
      "grade": "10th",
      "school": "Example School",
      "is_active": true,
      "created_at": "2024-01-15T10:30:00Z",
      "last_login": "2024-01-20T14:25:00Z",
      "performance_summary": {
        "total_attempts": 45,
        "accuracy": 0.78,
        "avg_score": 78.5
      }
    }
  }
}
```

**Authentication:** Admin JWT required

---

### PUT /api/admin/students/{student_id}
**Update student details**

**Path Parameters:**
- `student_id`: Student ID or MongoDB ObjectId

**Request Body:**
```json
{
  "name": "John Smith",
  "email": "johnsmith@example.com",
  "grade": "11th",
  "school": "New School"
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Student updated successfully",
  "data": {
    "student": {
      "student_id": "STU001",
      "name": "John Smith",
      "email": "johnsmith@example.com",
      "grade": "11th",
      "school": "New School",
      "is_active": true
    }
  }
}
```

**Authentication:** Admin JWT required

---

### DELETE /api/admin/students/{student_id}
**Soft delete (deactivate) a student**

**Path Parameters:**
- `student_id`: Student ID or MongoDB ObjectId

**Response (Success):**
```json
{
  "success": true,
  "message": "Student deleted successfully"
}
```

**Authentication:** Admin JWT required

---

### POST /api/admin/students/{student_id}/reset-password
**Reset student password**

**Path Parameters:**
- `student_id`: Student ID or MongoDB ObjectId

**Request Body:**
```json
{
  "new_password": "newsecurepass123"
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Password reset successfully"
}
```

**Authentication:** Admin JWT required

---

### GET /api/admin/students/{student_id}/performance
**Get detailed performance analytics for a student**

**Path Parameters:**
- `student_id`: Student ID or MongoDB ObjectId

**Response (Success):**
```json
{
  "success": true,
  "message": "Student performance retrieved successfully",
  "data": {
    "student": {
      "student_id": "STU001",
      "name": "John Doe",
      "email": "john@example.com"
    },
    "performance": {
      "total_attempts": 45,
      "accuracy": 0.78,
      "avg_score": 78.5,
      "difficulty_breakdown": {
        "easy": {"attempts": 15, "accuracy": 0.87},
        "medium": {"attempts": 20, "accuracy": 0.75},
        "hard": {"attempts": 10, "accuracy": 0.65}
      },
      "subject_performance": {
        "Mathematics": {"attempts": 20, "accuracy": 0.82},
        "Physics": {"attempts": 15, "accuracy": 0.73},
        "Chemistry": {"attempts": 10, "accuracy": 0.71}
      },
      "recent_activity": [
        {
          "date": "2024-01-20",
          "attempts": 5,
          "accuracy": 0.8,
          "avg_score": 80
        }
      ]
    }
  }
}
```

**Authentication:** Admin JWT required

---

### GET /api/admin/students/{student_id}/attempts
**Get student's question attempts with pagination**

**Path Parameters:**
- `student_id`: Student ID or MongoDB ObjectId

**Query Parameters:**
- `page` (int, default: 1): Page number
- `limit` (int, default: 50, max: 100): Items per page

**Response (Success):**
```json
{
  "success": true,
  "message": "Student attempts retrieved successfully",
  "data": {
    "student": {
      "student_id": "STU001",
      "name": "John Doe"
    },
    "attempts": [
      {
        "question_id": "q123",
        "question_text": "What is 2+2?",
        "is_correct": true,
        "score": 100,
        "time_taken": 30,
        "created_at": "2024-01-20T14:25:00Z",
        "subject": "Mathematics",
        "difficulty": "easy"
      }
    ],
    "total": 45,
    "page": 1,
    "limit": 50
  }
}
```

**Authentication:** Admin JWT required

---

### GET /api/admin/analytics/overview
**Get comprehensive analytics overview for all students**

**Response (Success):**
```json
{
  "success": true,
  "message": "Analytics overview retrieved successfully",
  "data": {
    "students_performance": [
      {
        "student_id": "STU001",
        "name": "John Doe",
        "total_attempts": 45,
        "accuracy": 0.78,
        "avg_score": 78.5,
        "last_activity": "2024-01-20T14:25:00Z"
      }
    ],
    "summary": {
      "total_students": 25,
      "total_attempts": 1250,
      "avg_accuracy": 0.76
    }
  }
}
```

**Authentication:** Admin JWT required

---

### GET /api/admin/dashboard/stats
**Get admin dashboard statistics**

**Response (Success):**
```json
{
  "success": true,
  "message": "Dashboard stats retrieved successfully",
  "data": {
    "total_students": 25,
    "total_attempts": 1250,
    "recent_students": 3,
    "recent_attempts": 45,
    "top_students": [
      {
        "student_id": "STU001",
        "name": "John Doe",
        "accuracy": 0.85,
        "total_attempts": 60
      }
    ]
  }
}
```

**Authentication:** Admin JWT required

---

## Student Endpoints

### GET /api/student/profile
**Get student profile information**

**Response (Success):**
```json
{
  "success": true,
  "message": "Profile retrieved successfully",
  "data": {
    "student": {
      "student_id": "STU001",
      "username": "student001",
      "name": "John Doe",
      "email": "john@example.com",
      "grade": "10th",
      "school": "Example School",
      "is_active": true,
      "created_at": "2024-01-15T10:30:00Z",
      "last_login": "2024-01-20T14:25:00Z"
    }
  }
}
```

**Authentication:** Student JWT required

---

### POST /api/student/attempts
**Submit a question attempt**

**Request Body:**
```json
{
  "question_id": "q123",
  "question_type": "mcq",
  "question_text": "What is 2+2?",
  "student_answer": "4",
  "correct_answer": "4",
  "is_correct": true,
  "score": 100,
  "time_taken": 30,
  "hints_used": 0,
  "mode": "practice",
  "difficulty": "easy",
  "subject": "Mathematics",
  "topic": "Basic Arithmetic"
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Attempt submitted successfully",
  "data": {
    "attempt_id": "507f1f77bcf86cd799439011",
    "attempt": {
      "question_id": "q123",
      "student_answer": "4",
      "is_correct": true,
      "score": 100,
      "created_at": "2024-01-20T14:30:00Z"
    }
  }
}
```

**Authentication:** Student JWT required

---

### GET /api/student/attempts
**Get student's own question attempts with pagination**

**Query Parameters:**
- `page` (int, default: 1): Page number
- `limit` (int, default: 50, max: 100): Items per page

**Response (Success):**
```json
{
  "success": true,
  "message": "Attempts retrieved successfully",
  "data": {
    "attempts": [
      {
        "question_id": "q123",
        "question_text": "What is 2+2?",
        "student_answer": "4",
        "is_correct": true,
        "score": 100,
        "time_taken": 30,
        "created_at": "2024-01-20T14:30:00Z",
        "subject": "Mathematics",
        "difficulty": "easy"
      }
    ],
    "total": 45,
    "page": 1,
    "limit": 50
  }
}
```

**Authentication:** Student JWT required

---

### GET /api/student/performance
**Get student's own performance analytics**

**Response (Success):**
```json
{
  "success": true,
  "message": "Performance retrieved successfully",
  "data": {
    "performance": {
      "total_attempts": 45,
      "accuracy": 0.78,
      "avg_score": 78.5,
      "difficulty_breakdown": {
        "easy": {"attempts": 15, "accuracy": 0.87},
        "medium": {"attempts": 20, "accuracy": 0.75},
        "hard": {"attempts": 10, "accuracy": 0.65}
      },
      "subject_performance": {
        "Mathematics": {"attempts": 20, "accuracy": 0.82},
        "Physics": {"attempts": 15, "accuracy": 0.73},
        "Chemistry": {"attempts": 10, "accuracy": 0.71}
      }
    }
  }
}
```

**Authentication:** Student JWT required

---

### GET /api/student/dashboard/stats
**Get dashboard statistics for student**

**Response (Success):**
```json
{
  "success": true,
  "message": "Dashboard stats retrieved successfully",
  "data": {
    "total_attempts": 45,
    "accuracy": 0.78,
    "avg_score": 78.5,
    "recent_attempts": 5,
    "current_streak": 3,
    "difficulty_breakdown": {
      "easy": {"attempts": 15, "accuracy": 0.87},
      "medium": {"attempts": 20, "accuracy": 0.75},
      "hard": {"attempts": 10, "accuracy": 0.65}
    },
    "subject_performance": {
      "Mathematics": {"attempts": 20, "accuracy": 0.82},
      "Physics": {"attempts": 15, "accuracy": 0.73},
      "Chemistry": {"attempts": 10, "accuracy": 0.71}
    }
  }
}
```

**Authentication:** Student JWT required

---

### POST /api/student/practice/evaluate
**Evaluate a practice question answer**

**Request Body:**
```json
{
  "question_id": "q123",
  "question_text": "Solve for x: 2x + 3 = 7",
  "student_answer": "x = 2",
  "correct_answer": "x = 2",
  "question_type": "practice",
  "time_taken": 120
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Answer evaluated successfully",
  "data": {
    "is_correct": true,
    "score": 100,
    "feedback": "Excellent work! You correctly solved the equation.",
    "correct_answer": "x = 2"
  }
}
```

**Authentication:** Student JWT required

---

### POST /api/student/change-password
**Change student password**

**Request Body:**
```json
{
  "current_password": "oldpassword123",
  "new_password": "newpassword123"
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Password changed successfully"
}
```

**Authentication:** Student JWT required

---

## Question Management Endpoints

### GET /api/questions/{question_id}
**Get a specific question by ID**

**Path Parameters:**
- `question_id`: Question ID

**Query Parameters:**
- `include_images` (boolean, default: true): Include base64 image data

**Response (Success):**
```json
{
  "success": true,
  "question": {
    "id": "q123",
    "text": "What is the capital of France?",
    "subject": "Geography",
    "difficulty": "easy",
    "options": ["London", "Paris", "Berlin", "Madrid"],
    "correctAnswer": "Paris",
    "images": [
      {
        "id": "img1",
        "filename": "paris_map.jpg",
        "path": "images/paris_map.jpg",
        "description": "Map of France",
        "type": "diagram"
      }
    ],
    "metadata": {
      "topic": "European Capitals",
      "grade_level": "6-8"
    }
  }
}
```

**Authentication:** None required

---

### GET /api/questions
**Get questions with optional filtering**

**Query Parameters:**
- `limit` (int, default: 50): Maximum number of questions
- `subject` (string, optional): Filter by subject
- `difficulty` (string, optional): Filter by difficulty (easy/medium/hard)
- `include_images` (boolean, default: true): Include base64 image data

**Response (Success):**
```json
{
  "success": true,
  "questions": [
    {
      "id": "q123",
      "text": "What is the capital of France?",
      "subject": "Geography",
      "difficulty": "easy",
      "options": ["London", "Paris", "Berlin", "Madrid"],
      "images": []
    }
  ],
  "count": 25
}
```

**Authentication:** None required

---

### GET /api/questions/search
**Search questions with filters**

**Query Parameters:**
- `query` (string, optional): Search text
- `subject` (string, optional): Filter by subject
- `difficulty` (string, optional): Filter by difficulty
- `has_images` (boolean, optional): Filter by presence of images
- `limit` (int, default: 50, max: 100): Maximum results
- `include_images` (boolean, default: false): Include base64 image data

**Response (Success):**
```json
{
  "success": true,
  "questions": [
    {
      "id": "q123",
      "text": "What is the capital of France?",
      "subject": "Geography",
      "difficulty": "easy",
      "images": []
    }
  ],
  "count": 1
}
```

**Authentication:** None required

---

### POST /api/questions/save
**Save a single question to MongoDB**

**Request Body:**
```json
{
  "id": "q123",
  "text": "What is the capital of France?",
  "subject": "Geography",
  "difficulty": "easy",
  "options": ["London", "Paris", "Berlin", "Madrid"],
  "correctAnswer": "Paris",
  "images": [
    {
      "id": "img1",
      "filename": "map.jpg",
      "path": "images/map.jpg",
      "description": "Map of France",
      "type": "diagram",
      "base64Data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
    }
  ],
  "metadata": {
    "topic": "European Capitals",
    "grade_level": "6-8"
  }
}
```

**Response (Success):**
```json
{
  "success": true,
  "question_id": "q123",
  "message": "Question saved successfully"
}
```

**Authentication:** None required

---

### POST /api/questions/batch-save
**Save multiple questions to MongoDB**

**Request Body:**
```json
[
  {
    "id": "q123",
    "text": "What is the capital of France?",
    "subject": "Geography",
    "difficulty": "easy",
    "options": ["London", "Paris", "Berlin", "Madrid"],
    "correctAnswer": "Paris"
  },
  {
    "id": "q124",
    "text": "What is 2+2?",
    "subject": "Mathematics",
    "difficulty": "easy",
    "correctAnswer": "4"
  }
]
```

**Response (Success):**
```json
{
  "success": true,
  "success_count": 2,
  "total_count": 2,
  "message": "Saved 2 out of 2 questions"
}
```

**Authentication:** None required

---

### PUT /api/questions/{question_id}
**Update an existing question**

**Path Parameters:**
- `question_id`: Question ID

**Request Body:**
```json
{
  "text": "Updated question text",
  "difficulty": "medium",
  "metadata": {
    "topic": "Updated Topic"
  }
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Question updated successfully"
}
```

**Authentication:** None required

---

### DELETE /api/questions/{question_id}
**Delete a question and its associated images**

**Path Parameters:**
- `question_id`: Question ID

**Response (Success):**
```json
{
  "success": true,
  "message": "Question deleted successfully"
}
```

**Authentication:** None required

---

### GET /api/questions/stats
**Get questions collection statistics**

**Response (Success):**
```json
{
  "success": true,
  "statistics": {
    "total_questions": 1250,
    "questions_by_subject": {
      "Mathematics": 400,
      "Physics": 350,
      "Chemistry": 300,
      "Biology": 200
    },
    "questions_by_difficulty": {
      "easy": 500,
      "medium": 500,
      "hard": 250
    },
    "questions_with_images": 450,
    "total_images": 850
  }
}
```

**Authentication:** None required

---

### GET /api/questions/export
**Export all questions with images**

**Query Parameters:**
- `include_images` (boolean, default: false): Include base64 image data
- `subject` (string, optional): Filter by subject
- `difficulty` (string, optional): Filter by difficulty

**Response (Success):**
```json
{
  "success": true,
  "questions": [
    {
      "id": "q123",
      "text": "What is the capital of France?",
      "subject": "Geography",
      "difficulty": "easy",
      "images": []
    }
  ],
  "count": 1250,
  "exported_at": "2024-01-20T15:30:00Z"
}
```

**Authentication:** None required

---

## Image Management Endpoints

### POST /api/images/upload
**Upload an image file**

**Request Body:** Form-data with file
- `file`: Image file (JPEG, PNG, etc.)

**Response (Success):**
```json
{
  "success": true,
  "file_path": "images/uploaded_image.jpg",
  "filename": "uploaded_image.jpg",
  "image_info": {
    "width": 800,
    "height": 600,
    "format": "JPEG",
    "size_bytes": 125000
  },
  "message": "Image uploaded successfully"
}
```

**Authentication:** None required

---

### POST /api/images/upload-base64
**Upload an image from base64 data**

**Request Body:**
```json
{
  "base64Data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "filename": "diagram.jpg"
}
```

**Response (Success):**
```json
{
  "success": true,
  "file_path": "images/diagram.jpg",
  "filename": "diagram.jpg",
  "image_info": {
    "width": 800,
    "height": 600,
    "format": "JPEG",
    "size_bytes": 125000
  },
  "message": "Image uploaded successfully"
}
```

**Authentication:** None required

---

### GET /api/images/{path}
**Serve a stored image file**

**Path Parameters:**
- `path`: Image path (e.g., "images/diagram.jpg")

**Response:** Binary image data

**Authentication:** None required

---

### GET /api/images/{path}/base64
**Get image as base64 data**

**Path Parameters:**
- `path`: Image path (e.g., "images/diagram.jpg")

**Response (Success):**
```json
{
  "success": true,
  "base64Data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "image_path": "images/diagram.jpg"
}
```

**Authentication:** None required

---

### GET /api/images/{path}/info
**Get information about a stored image**

**Path Parameters:**
- `path`: Image path (e.g., "images/diagram.jpg")

**Response (Success):**
```json
{
  "success": true,
  "image_info": {
    "width": 800,
    "height": 600,
    "format": "JPEG",
    "size_bytes": 125000,
    "created_at": "2024-01-20T10:30:00Z"
  },
  "image_path": "images/diagram.jpg"
}
```

**Authentication:** None required

---

### DELETE /api/images/{path}
**Delete a stored image**

**Path Parameters:**
- `path`: Image path (e.g., "images/diagram.jpg")

**Response (Success):**
```json
{
  "success": true,
  "message": "Image deleted successfully"
}
```

**Authentication:** None required

---

## Chat Endpoints

### POST /api/chat
**Send message and get AI response**

**Request Body:**
```json
{
  "message": "Explain Newton's laws of motion",
  "sessionId": "session123",
  "userId": "user456",
  "mode": "general",
  "conversationHistory": [
    {
      "role": "user",
      "content": "What is physics?",
      "timestamp": "2024-01-20T10:00:00Z"
    },
    {
      "role": "assistant",
      "content": "Physics is the study of matter and energy...",
      "timestamp": "2024-01-20T10:00:05Z"
    }
  ],
  "canvasData": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "subject": "Physics"
}
```

**Response (Success):**
```json
{
  "success": true,
  "data": {
    "response": "Newton's laws of motion are three fundamental principles...",
    "usage": {
      "prompt_tokens": 150,
      "completion_tokens": 200,
      "total_tokens": 350
    },
    "model": "gpt-4",
    "sessionId": "session123",
    "mode": "general"
  }
}
```

**Authentication:** None required

---

### GET /api/chat/health
**Check chat service health**

**Response (Success):**
```json
{
  "success": true,
  "message": "Chat service is healthy",
  "model": "gpt-4",
  "service": "openai"
}
```

**Authentication:** None required

---

### GET /api/chat/models
**Get available AI models**

**Response (Success):**
```json
{
  "success": true,
  "data": {
    "current_model": "gpt-4",
    "provider": "openai",
    "system_prompts": ["general", "whiteboard", "practice", "mock-test"]
  }
}
```

**Authentication:** None required

---

## Practice Mode Endpoints

### POST /api/practice/next
**Get next practice question from MongoDB**

**Request Body:**
```json
{
  "subject": "Mathematics",
  "difficulty": "medium",
  "excludeIds": ["q123", "q124"],
  "document_id": "doc456"
}
```

**Response (Success):**
```json
{
  "success": true,
  "question": {
    "id": "q125",
    "text": "Solve for x: 2x² + 3x - 5 = 0",
    "subject": "Mathematics",
    "difficulty": "medium",
    "images": [
      {
        "id": "img1",
        "filename": "quadratic.jpg",
        "path": "images/quadratic.jpg",
        "base64Data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
      }
    ]
  }
}
```

**Authentication:** None required

---

### POST /api/practice/evaluate
**Evaluate canvas/text answer against ground-truth**

**Request Body:**
```json
{
  "questionId": "q125",
  "answerText": "x = (-3 ± √(9 + 40))/4",
  "canvasData": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "timeTaken": 180
}
```

**Response (Success):**
```json
{
  "success": true,
  "evaluation": {
    "correct": true,
    "score": 0.95,
    "extractedAnswer": "x = (-3 ± √49)/4 = (-3 ± 7)/4",
    "feedback": "Excellent work! You correctly applied the quadratic formula.",
    "reasoning": "Student showed all steps and arrived at the correct answer.",
    "extractedContent": "x = (-3 ± √(9 + 40))/4 = (-3 ± √49)/4 = (-3 ± 7)/4",
    "analysisType": "comprehensive_solution"
  },
  "question": {
    "id": "q125",
    "text": "Solve for x: 2x² + 3x - 5 = 0",
    "correctAnswer": "x = (-3 ± √49)/4"
  }
}
```

**Authentication:** JWT optional (for tracking)

---

## MCQ Mode Endpoints

### POST /api/mcq/check
**Check MCQ answer and get solution**

**Request Body:**
```json
{
  "question_id": "q123",
  "selected_answer": "B",
  "time_taken": 45
}
```

**Response (Success):**
```json
{
  "success": true,
  "result": {
    "is_correct": true,
    "correct_answer": "B",
    "solution": {
      "question_id": "q123",
      "correct_answer": "B",
      "explanation": "The answer is B because...",
      "detailed_solution": "Step-by-step explanation here...",
      "hints": ["Hint 1", "Hint 2"]
    },
    "feedback": "Correct! Well done.",
    "points": 10
  }
}
```

**Authentication:** JWT optional (for tracking)

---

### GET /api/mcq/solution/{question_id}
**Get stored solution for question**

**Path Parameters:**
- `question_id`: Question ID

**Response (Success):**
```json
{
  "success": true,
  "solution": {
    "id": "sol123",
    "question_id": "q123",
    "correct_answer": "B",
    "explanation": "The answer is B because...",
    "detailed_solution": "Step-by-step explanation here...",
    "hints": ["Hint 1", "Hint 2"],
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

**Authentication:** None required

---

### GET /api/mcq/random-question
**Get random MCQ question**

**Query Parameters:**
- `subject` (string, optional): Filter by subject
- `difficulty` (string, optional): Filter by difficulty

**Response (Success):**
```json
{
  "success": true,
  "question": {
    "id": "q123",
    "text": "What is the capital of France?",
    "subject": "Geography",
    "difficulty": "easy",
    "options": ["London", "Paris", "Berlin", "Madrid"],
    "enhancedOptions": [
      {"id": "A", "text": "London", "image": null},
      {"id": "B", "text": "Paris", "image": null},
      {"id": "C", "text": "Berlin", "image": null},
      {"id": "D", "text": "Madrid", "image": null}
    ],
    "images": []
  }
}
```

**Authentication:** None required

---

### POST /api/mcq/solution
**Save solution manually**

**Request Body:**
```json
{
  "id": "sol123",
  "question_id": "q123",
  "correct_answer": "B",
  "explanation": "The answer is B because...",
  "detailed_solution": "Step-by-step explanation here...",
  "hints": ["Hint 1", "Hint 2"],
  "generated_by": "admin"
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Solution saved successfully"
}
```

**Authentication:** None required

---

### GET /api/mcq/stats
**Get MCQ solutions statistics**

**Response (Success):**
```json
{
  "success": true,
  "statistics": {
    "total_solutions": 500,
    "solutions_by_subject": {
      "Mathematics": 150,
      "Physics": 120,
      "Chemistry": 130,
      "Biology": 100
    },
    "recent_solutions": 25
  }
}
```

**Authentication:** None required

---

## Debugger Mode Endpoints

### POST /api/debugger/chat
**Handle debugger chat message with conversation memory**

**Request Body:**
```json
{
  "sessionId": "debug_session_123",
  "message": "How do I fix this Python error?",
  "attachments": ["https://example.com/code.png"],
  "imageData": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**Response (Success):**
```json
{
  "success": true,
  "data": {
    "user_message": {
      "id": "msg123",
      "role": "user",
      "content": "How do I fix this Python error?",
      "timestamp": "2024-01-20T15:30:00Z"
    },
    "assistant_message": {
      "id": "msg124",
      "role": "assistant",
      "content": "The error suggests you have a syntax error...",
      "timestamp": "2024-01-20T15:30:02Z"
    },
    "response": "The error suggests you have a syntax error...",
    "session_id": "debug_session_123",
    "message_count": 5,
    "usage": {
      "prompt_tokens": 200,
      "completion_tokens": 150,
      "total_tokens": 350
    },
    "model": "gpt-4"
  }
}
```

**Authentication:** None required

---

### GET /api/debugger/history/{session_id}
**Get conversation history for a session**

**Path Parameters:**
- `session_id`: Session ID

**Query Parameters:**
- `limit` (int, optional): Maximum number of messages

**Response (Success):**
```json
{
  "success": true,
  "data": {
    "session_id": "debug_session_123",
    "messages": [
      {
        "id": "msg123",
        "role": "user",
        "content": "How do I fix this Python error?",
        "timestamp": "2024-01-20T15:30:00Z"
      },
      {
        "id": "msg124",
        "role": "assistant",
        "content": "The error suggests you have a syntax error...",
        "timestamp": "2024-01-20T15:30:02Z"
      }
    ],
    "metadata": {
      "created_at": "2024-01-20T15:25:00Z",
      "last_activity": "2024-01-20T15:30:02Z",
      "total_messages": 5
    }
  }
}
```

**Authentication:** None required

---

### POST /api/debugger/session/{session_id}/clear
**Clear conversation history for a session**

**Path Parameters:**
- `session_id`: Session ID

**Response (Success):**
```json
{
  "success": true,
  "message": "Session cleared successfully"
}
```

**Authentication:** None required

---

### DELETE /api/debugger/session/{session_id}
**Delete a conversation session**

**Path Parameters:**
- `session_id`: Session ID

**Response (Success):**
```json
{
  "success": true,
  "message": "Session deleted successfully"
}
```

**Authentication:** None required

---

### GET /api/debugger/sessions
**Get statistics about all active sessions**

**Response (Success):**
```json
{
  "success": true,
  "data": {
    "active_sessions": 15,
    "sessions": [
      {
        "session_id": "debug_session_123",
        "created_at": "2024-01-20T15:25:00Z",
        "last_activity": "2024-01-20T15:30:02Z",
        "message_count": 5
      }
    ]
  }
}
```

**Authentication:** None required

---

## Health Check Endpoints

### GET /health
**Comprehensive health check with service status**

**Response (Success):**
```json
{
  "success": true,
  "healthy": true,
  "ok": true,
  "status": "healthy",
  "message": "Backend server is running",
  "timestamp": 1705770000.123,
  "services": {
    "database": "healthy",
    "cache": "optional",
    "mongodb": {
      "connected": true,
      "status": "online",
      "questions_count": 1250
    }
  },
  "version": "2.0.0",
  "mode": "development"
}
```

**Authentication:** None required

---

## Data Models

### Student Model
```python
{
  "_id": ObjectId,  # MongoDB ObjectId
  "student_id": str,  # Unique student identifier (e.g., "STU001")
  "username": str,   # Login username
  "password_hash": str,  # Hashed password
  "name": str,       # Full name
  "email": str,      # Email address
  "grade": str,      # Grade level (e.g., "10th")
  "school": str,     # School name
  "phone": str,      # Phone number (optional)
  "is_active": bool, # Account status
  "created_at": datetime,
  "updated_at": datetime,
  "last_login": datetime,
  "created_by": ObjectId,  # Admin who created the account
  "admin_id": ObjectId     # Associated admin for data isolation
}
```

### Question Model
```python
{
  "id": str,           # Unique question identifier
  "text": str,         # Question text
  "subject": str,      # Subject (Mathematics, Physics, Chemistry, Biology)
  "difficulty": str,   # Difficulty level (easy, medium, hard)
  "extractedAt": str,  # ISO datetime string
  "pdfSource": str,    # Source PDF document
  "images": [          # Associated images
    {
      "id": str,
      "filename": str,
      "path": str,
      "description": str,
      "type": str,
      "base64Data": str,  # Only for transfer
      "bbox": dict,       # Bounding box coordinates
      "metadata": dict
    }
  ],
  "options": list,     # MCQ options (optional)
  "correctAnswer": str,# Correct answer (optional)
  "metadata": dict     # Additional metadata
}
```

### QuestionAttempt Model
```python
{
  "_id": ObjectId,
  "student_id": str,     # Student identifier
  "question_id": str,    # Question identifier
  "question_type": str,  # Type (mcq, practice, etc.)
  "question_text": str,  # Question content
  "student_answer": str, # Student's answer
  "correct_answer": str, # Correct answer
  "is_correct": bool,    # Whether answer was correct
  "score": float,        # Score (0-100)
  "time_taken": int,     # Time in seconds
  "hints_used": int,     # Number of hints used
  "mode": str,           # Mode (practice, mcq, etc.)
  "difficulty": str,     # Question difficulty
  "subject": str,        # Subject
  "topic": str,          # Topic within subject
  "created_at": datetime,
  "admin_id": ObjectId   # For data isolation
}
```

## Error Handling

### Standard Error Response Format
```json
{
  "success": false,
  "error": "error_type",
  "message": "Human-readable error message",
  "details": {}  // Optional additional error details
}
```

### Common HTTP Status Codes
- `200`: Success
- `201`: Created
- `400`: Bad Request (validation errors)
- `401`: Unauthorized (invalid/missing JWT)
- `403`: Forbidden (insufficient permissions)
- `404`: Not Found
- `405`: Method Not Allowed
- `409`: Conflict (duplicate data)
- `422`: Unprocessable Entity (validation failed)
- `500`: Internal Server Error
- `502`: Bad Gateway (external service error)
- `503`: Service Unavailable

### Authentication Errors
```json
{
  "success": false,
  "error": "Unauthorized",
  "message": "JWT token is required"
}
```

### Validation Errors
```json
{
  "success": false,
  "error": "Validation error",
  "message": "Invalid input data",
  "details": {
    "field_name": ["Error message 1", "Error message 2"]
  }
}
```

---

## Additional Notes

1. **JWT Tokens**: All authenticated endpoints require a valid JWT token in the Authorization header: `Authorization: Bearer <token>`

2. **Data Isolation**: Admin endpoints are isolated by `admin_id` to support multi-tenant deployments

3. **Image Handling**: Images are stored in the `images/` directory and served via dedicated endpoints

4. **MongoDB**: Question storage and retrieval uses MongoDB with flexible queries

5. **Rate Limiting**: Consider implementing rate limiting for chat and evaluation endpoints in production

6. **File Uploads**: Image uploads are limited to common formats (JPEG, PNG, GIF) with size restrictions

7. **Session Management**: Chat sessions are maintained in memory with configurable expiration

8. **Logging**: All endpoints include comprehensive logging for debugging and monitoring

This documentation provides a complete overview of the SkillBot Backend API. For development and testing, the server runs on port 5001 by default.
