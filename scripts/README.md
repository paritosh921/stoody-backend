# Backend Utility Scripts

This directory contains utility scripts for database management, admin operations, and migrations.

## Directory Structure

```
scripts/
├── admin/          # Admin account management utilities
├── database/       # Database maintenance and sync utilities
└── migrations/     # One-time migration scripts
```

---

## Admin Scripts (`admin/`)

### `init_admin_direct.py`
**Purpose**: Initialize or reset the default admin account directly in MongoDB

**Usage**:
```bash
cd backend
python scripts/admin/init_admin_direct.py
```

**What it does**:
- Creates a default admin account with:
  - Email: `admin@skillbot.app`
  - Password: `admin123`
  - Subdomain: `main`
- Useful for initial setup or recovery

**⚠️ Warning**: Only use in development or for account recovery

---

### `update_admin_password.py`
**Purpose**: Reset an admin account password

**Usage**:
```bash
cd backend
python scripts/admin/update_admin_password.py
```

**What it does**:
- Prompts for admin email
- Prompts for new password
- Updates password_hash in MongoDB
- Useful for password recovery

**Use Case**: Admin forgot password and needs manual reset

---

## Database Scripts (`database/`)

### `check_mongodb_questions.py`
**Purpose**: Check and display questions stored in MongoDB

**Usage**:
```bash
cd backend
python scripts/database/check_mongodb_questions.py
```

**What it does**:
- Connects to MongoDB
- Lists all questions in `questions` collection
- Shows document counts and structure
- Useful for debugging MongoDB data

---

### `check_documents.py`
**Purpose**: Check PDF documents stored in MongoDB

**Usage**:
```bash
cd backend
python scripts/database/check_documents.py
```

**What it does**:
- Lists all PDF documents in MongoDB
- Shows document metadata (type, standards, subjects, etc.)
- Counts questions per document
- Useful for auditing uploaded content

---

---

## Migration Scripts (`migrations/`)

### `fix_question_document_types.py`
**Purpose**: One-time migration to fix question document_type field

**Usage**:
```bash
cd backend
python scripts/migrations/fix_question_document_types.py
```

**What it does**:
- Updates questions with missing or incorrect `document_type` field
- Maps questions to their parent document's type
- Ensures data consistency

**When to use**:
- After schema changes
- One-time migration (likely already run)

**Note**: Safe to run multiple times (idempotent)

---

## Common Workflows

### Initial Setup
```bash
# 1. Initialize default admin
python scripts/admin/init_admin_direct.py

# 2. Check if admin was created
mongosh "your_mongodb_uri" --eval "db.admins.find()"
```

### After Uploading PDFs
```bash
# 1. Check documents were created
python scripts/database/check_documents.py

# 2. Check questions were extracted
python scripts/database/check_mongodb_questions.py
```

### Complete Database Reset
```bash
# ⚠️ WARNING: This deletes all questions!

# 1. Delete questions from MongoDB
mongosh "your_mongodb_uri" --eval "db.questions.deleteMany({})"

# 2. Re-upload PDFs via admin panel
```

### Password Recovery
```bash
# Reset admin password
python scripts/admin/update_admin_password.py

# Follow prompts to enter email and new password
```

---

## Dependencies

All scripts require:
- Python 3.8+
- MongoDB connection (`.env` file configured)
- Installed packages from `requirements.txt`

To install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

---

## Environment Variables

Scripts use environment variables from `backend/.env`:

```env
MONGODB_URI=mongodb+srv://...
MONGODB_DB_NAME=skillbot_db
```

Ensure `.env` file is properly configured before running scripts.

---

## Safety Notes

### Safe Scripts (Read-Only)
✅ `check_mongodb_questions.py`
✅ `check_documents.py`

### Moderate Risk (Write Operations)
⚠️ `init_admin_direct.py` - Creates admin account
⚠️ `update_admin_password.py` - Modifies admin password

---

## Troubleshooting

### Script fails with "ModuleNotFoundError"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Script fails with "MongoDB connection error"
**Solution**: Check `.env` file has correct `MONGODB_URI`

### "Permission denied" error
**Solution**: Ensure you're running from backend directory:
```bash
cd backend
python scripts/database/script_name.py
```

---

## Future Improvements

- [ ] Add interactive CLI for admin scripts
- [ ] Add backup/restore utilities
- [ ] Create student account bulk import script
- [ ] Add database health check script

---

**Last Updated**: October 2025
**Maintained By**: Stoody® Development Team
