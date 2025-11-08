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

### `check_chromadb_questions.py`
**Purpose**: Check and display questions stored in ChromaDB

**Usage**:
```bash
cd backend
python scripts/database/check_chromadb_questions.py
```

**What it does**:
- Connects to ChromaDB
- Lists all questions in the collection
- Shows metadata and counts
- Useful for debugging vector database

---

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

### `reset_chromadb.py`
**Purpose**: Delete all data from ChromaDB

**Usage**:
```bash
cd backend
python scripts/database/reset_chromadb.py
```

**What it does**:
- Deletes the entire ChromaDB collection
- Removes all vector embeddings
- Clears question data from ChromaDB

**⚠️ DANGER**: This is destructive! Only use when you want to completely reset the vector database.

**When to use**:
- Before re-syncing all questions
- When ChromaDB is corrupted
- During testing/development

---

### `sync_chromadb.py`
**Purpose**: Sync questions from MongoDB to ChromaDB

**Usage**:
```bash
cd backend
python scripts/database/sync_chromadb.py
```

**What it does**:
- Reads questions from MongoDB
- Creates vector embeddings
- Stores in ChromaDB for semantic search
- Updates existing questions if already synced

**When to use**:
- After adding new questions via PDF upload
- After database restoration
- When ChromaDB is out of sync with MongoDB

---

### `sync_questions_to_chromadb.py`
**Purpose**: Alternative sync script with more detailed logging

**Usage**:
```bash
cd backend
python scripts/database/sync_questions_to_chromadb.py
```

**What it does**:
- Similar to `sync_chromadb.py`
- More verbose output
- Shows progress per question

**Note**: Consider merging with `sync_chromadb.py` in the future

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

# 3. Sync to ChromaDB for semantic search
python scripts/database/sync_chromadb.py

# 4. Verify ChromaDB has questions
python scripts/database/check_chromadb_questions.py
```

### Database Maintenance
```bash
# Check MongoDB and ChromaDB are in sync
python scripts/database/check_mongodb_questions.py
python scripts/database/check_chromadb_questions.py

# If counts don't match, re-sync
python scripts/database/sync_chromadb.py
```

### Complete Database Reset
```bash
# ⚠️ WARNING: This deletes all questions!

# 1. Reset ChromaDB
python scripts/database/reset_chromadb.py

# 2. Delete questions from MongoDB
mongosh "your_mongodb_uri" --eval "db.questions.deleteMany({})"

# 3. Re-upload PDFs via admin panel
# 4. Sync to ChromaDB
python scripts/database/sync_chromadb.py
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
CHROMADB_PATH=./chromadb_data
```

Ensure `.env` file is properly configured before running scripts.

---

## Safety Notes

### Safe Scripts (Read-Only)
✅ `check_chromadb_questions.py`
✅ `check_mongodb_questions.py`
✅ `check_documents.py`

### Moderate Risk (Write Operations)
⚠️ `init_admin_direct.py` - Creates admin account
⚠️ `update_admin_password.py` - Modifies admin password
⚠️ `sync_chromadb.py` - Writes to ChromaDB
⚠️ `sync_questions_to_chromadb.py` - Writes to ChromaDB
⚠️ `fix_question_document_types.py` - Modifies MongoDB

### High Risk (Destructive)
🚨 `reset_chromadb.py` - DELETES all ChromaDB data

---

## Troubleshooting

### Script fails with "ModuleNotFoundError"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Script fails with "MongoDB connection error"
**Solution**: Check `.env` file has correct `MONGODB_URI`

### ChromaDB scripts fail with "Collection not found"
**Solution**: ChromaDB may be empty. Run sync script:
```bash
python scripts/database/sync_chromadb.py
```

### "Permission denied" error
**Solution**: Ensure you're running from backend directory:
```bash
cd backend
python scripts/database/script_name.py
```

---

## Future Improvements

- [ ] Consolidate `sync_chromadb.py` and `sync_questions_to_chromadb.py`
- [ ] Add interactive CLI for admin scripts
- [ ] Add backup/restore utilities
- [ ] Create student account bulk import script
- [ ] Add database health check script

---

**Last Updated**: October 2025
**Maintained By**: Stoody® Development Team
