# SkillBot Backend Scripts - Complete Index

## 📋 Quick Navigation

| Script | Platform | Purpose | When to Use |
|--------|----------|---------|-------------|
| [run_backend.bat](#run_backendbat) | Windows | Start server | Every time you want to run the backend |
| [run_backend.sh](#run_backendsh) | Linux/Mac | Start server | Every time you want to run the backend |
| [setup_backend.bat](#setup_backendbat) | Windows | Initial setup | First time only, or when dependencies change |
| [setup_backend.sh](#setup_backendsh) | Linux/Mac | Initial setup | First time only, or when dependencies change |
| [test_setup.bat](#test_setupbat) | Windows | Verify setup | After running setup, before first run |
| [test_setup.sh](#test_setupsh) | Linux/Mac | Verify setup | After running setup, before first run |

## 📚 Documentation Files

| File | Purpose | When to Read |
|------|---------|--------------|
| [QUICK_START.md](#quick_startmd) | Quick reference | When you need a fast reminder |
| [BACKEND_SETUP.md](#backend_setupmd) | Comprehensive guide | When setting up or troubleshooting |
| [SCRIPTS_SUMMARY.md](#scripts_summarymd) | Scripts overview | When learning about the scripts |
| [SCRIPTS_INDEX.md](#scripts_indexmd) | This file | For navigation and reference |

---

## 🚀 Getting Started

### Windows Users

```batch
# Step 1: Setup (first time only)
setup_backend.bat

# Step 2: Verify setup
test_setup.bat

# Step 3: Run server
run_backend.bat
```

### Linux/Mac Users

```bash
# Step 0: Make scripts executable
chmod +x *.sh

# Step 1: Setup (first time only)
./setup_backend.sh

# Step 2: Verify setup
./test_setup.sh

# Step 3: Run server
./run_backend.sh
```

---

## 📖 Script Details

### run_backend.bat
**Platform**: Windows
**Purpose**: Start the FastAPI backend server
**Usage**: `run_backend.bat` or double-click

**What it does**:
1. Checks if virtual environment exists
2. Activates the virtual environment
3. Displays server configuration
4. Starts the server with `python main_async.py`
5. Shows errors if startup fails

**Requirements**:
- Virtual environment must exist (run `setup_backend.bat` first)
- `.env` file should be configured

**Output**:
- Server runs on http://localhost:5001
- API docs at http://localhost:5001/docs
- Press Ctrl+C to stop

---

### run_backend.sh
**Platform**: Linux/Mac
**Purpose**: Start the FastAPI backend server
**Usage**: `./run_backend.sh`

**What it does**:
1. Checks if virtual environment exists
2. Activates the virtual environment
3. Displays server configuration
4. Starts the server with `python main_async.py`
5. Shows errors if startup fails

**Requirements**:
- Virtual environment must exist (run `./setup_backend.sh` first)
- `.env` file should be configured
- Execute permission (`chmod +x run_backend.sh`)

**Output**:
- Server runs on http://localhost:5001
- API docs at http://localhost:5001/docs
- Press Ctrl+C to stop

---

### setup_backend.bat
**Platform**: Windows
**Purpose**: Complete backend environment setup
**Usage**: `setup_backend.bat` or double-click

**What it does**:
1. Checks Python installation
2. Creates virtual environment (or recreates if exists)
3. Activates virtual environment
4. Upgrades pip to latest version
5. Installs all dependencies from `requirements.txt`
6. Shows success message with next steps

**Time**: 5-10 minutes (depending on internet speed)

**Requirements**:
- Python 3.9+ installed
- Internet connection for package downloads
- `requirements.txt` in same directory

**When to run**:
- First time setup
- After cloning the repository
- When dependencies change
- If virtual environment is corrupted

---

### setup_backend.sh
**Platform**: Linux/Mac
**Purpose**: Complete backend environment setup
**Usage**: `./setup_backend.sh`

**What it does**:
1. Checks Python 3 installation
2. Creates virtual environment (or recreates if exists)
3. Activates virtual environment
4. Upgrades pip to latest version
5. Installs all dependencies from `requirements.txt`
6. Shows success message with next steps

**Time**: 5-10 minutes (depending on internet speed)

**Requirements**:
- Python 3.9+ installed
- Internet connection for package downloads
- `requirements.txt` in same directory
- Execute permission (`chmod +x setup_backend.sh`)

**When to run**:
- First time setup
- After cloning the repository
- When dependencies change
- If virtual environment is corrupted

---

### test_setup.bat
**Platform**: Windows
**Purpose**: Verify backend setup is complete and correct
**Usage**: `test_setup.bat` or double-click

**What it checks**:
1. Python installation
2. Virtual environment exists
3. FastAPI is installed
4. `.env` file exists
5. `main_async.py` exists

**Output**:
- Pass/Fail for each check
- Overall success or failure message
- Suggestions if checks fail

**When to run**:
- After running `setup_backend.bat`
- Before running server for the first time
- When troubleshooting issues
- To verify environment integrity

---

### test_setup.sh
**Platform**: Linux/Mac
**Purpose**: Verify backend setup is complete and correct
**Usage**: `./test_setup.sh`

**What it checks**:
1. Python 3 installation
2. Virtual environment exists
3. FastAPI is installed
4. `.env` file exists
5. `main_async.py` exists

**Output**:
- Pass/Fail for each check
- Overall success or failure message
- Suggestions if checks fail

**When to run**:
- After running `./setup_backend.sh`
- Before running server for the first time
- When troubleshooting issues
- To verify environment integrity

**Requirements**:
- Execute permission (`chmod +x test_setup.sh`)

---

## 📚 Documentation Details

### QUICK_START.md
**Purpose**: Minimal quick reference guide
**Best for**: Developers who need a quick reminder

**Contents**:
- One-line setup commands
- One-line run commands
- Access URLs
- Common issues and solutions
- What's running overview

---

### BACKEND_SETUP.md
**Purpose**: Comprehensive setup and configuration guide
**Best for**: First-time setup, troubleshooting, configuration

**Contents**:
- Prerequisites
- Step-by-step setup (Windows & Linux/Mac)
- Manual setup instructions
- Server configuration details
- Environment variables reference
- Troubleshooting guide
- Architecture overview
- API endpoints documentation
- Production deployment notes

---

### SCRIPTS_SUMMARY.md
**Purpose**: Overview of all scripts and their features
**Best for**: Understanding what each script does

**Contents**:
- Detailed description of each script
- Workflow diagrams
- Server details
- Key features
- Dependencies list
- Environment requirements
- Maintenance instructions

---

### SCRIPTS_INDEX.md
**Purpose**: This file - navigation hub for all documentation
**Best for**: Finding the right script or documentation

**Contents**:
- Quick navigation table
- Getting started workflows
- Detailed script descriptions
- Documentation summaries
- Common workflows
- Tips and tricks

---

## 🔄 Common Workflows

### First Time Setup
```
setup → test → run
```
1. Run setup script
2. Verify with test script
3. Start server with run script

### Daily Development
```
run
```
1. Just run the server
2. Code changes auto-reload

### Dependency Update
```
setup (recreate) → test → run
```
1. Run setup and choose to recreate venv
2. Verify with test script
3. Start server

### Troubleshooting
```
test → review docs → setup (if needed)
```
1. Run test script to identify issues
2. Review BACKEND_SETUP.md for solutions
3. Re-run setup if environment is corrupted

---

## 💡 Tips & Tricks

### Windows Tips
- Double-click `.bat` files to run them
- Keep terminal open to see server logs
- Use `Ctrl+C` to stop the server
- Check Task Manager if port 5001 is in use

### Linux/Mac Tips
- Always use `./script.sh` not just `script.sh`
- Make scripts executable once: `chmod +x *.sh`
- Use `ps aux | grep python` to find running processes
- Use `lsof -i :5001` to check what's using port 5001

### General Tips
- Run test script after any major changes
- Keep virtual environment separate from system Python
- Update `.env` before first run
- Check logs in `logs/app.log` for errors
- Use API docs at `/docs` for endpoint testing

---

## 🆘 Quick Troubleshooting

| Problem | Solution | Script to Run |
|---------|----------|---------------|
| "Python not found" | Install Python 3.9+ | N/A |
| "Virtual environment not found" | Run setup script | `setup_backend` |
| "Dependencies missing" | Run setup script | `setup_backend` |
| "Port 5001 in use" | Change PORT in `.env` or kill process | N/A |
| "MongoDB error" | Check MONGODB_URI or disable | N/A |
| "Server won't start" | Run test script, check logs | `test_setup` |

---

## 🎯 Next Steps

After successful setup:

1. **Configure Environment**
   - Edit `.env` with your API keys
   - Set MongoDB connection string
   - Configure Redis URL (optional in dev)

2. **Start Development**
   - Run `run_backend.bat` (Windows) or `./run_backend.sh` (Linux/Mac)
   - Visit http://localhost:5001/docs
   - Test endpoints with the interactive documentation

3. **Learn the API**
   - Read BACKEND_SETUP.md for architecture
   - Explore `/docs` for API reference
   - Check `/health` for service status

4. **Production Deployment**
   - Review production notes in BACKEND_SETUP.md
   - Configure environment for production
   - Set up process manager (PM2, systemd, etc.)
   - Enable Redis for caching and rate limiting

---

## 📞 Support

For help:
1. Check [BACKEND_SETUP.md](BACKEND_SETUP.md) troubleshooting section
2. Run `test_setup` script to diagnose issues
3. Review logs in `logs/app.log`
4. Check environment configuration in `.env`
5. Verify all services (MongoDB, Redis) are running

---

**Last Updated**: 2025-11-30
