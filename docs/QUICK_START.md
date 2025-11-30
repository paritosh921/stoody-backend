# SkillBot Backend - Quick Start

## Windows Users

### First Time Setup
```bash
setup_backend.bat
```

### Run Server
```bash
run_backend.bat
```

## Linux/Mac Users

### First Time Setup
```bash
chmod +x setup_backend.sh run_backend.sh
./setup_backend.sh
```

### Run Server
```bash
./run_backend.sh
```

## Access Points

- **API Docs**: http://localhost:5001/docs
- **Health Check**: http://localhost:5001/health
- **API Root**: http://localhost:5001/

## Stop Server

Press `Ctrl+C` in the terminal

## Troubleshooting

**Issue**: Virtual environment not found
**Solution**: Run the setup script first

**Issue**: Port already in use
**Solution**: Change `PORT` in `.env` or stop the process using port 5001

**Issue**: Dependencies fail to install
**Solution**:
1. Upgrade pip: `python -m pip install --upgrade pip`
2. Check Python version (must be 3.9+)

**Issue**: MongoDB connection error
**Solution**: Check `MONGODB_URI` in `.env` or set `DISABLE_MONGODB=true` for development

## What's Running?

The backend server runs:
- FastAPI application on port 5001
- Auto-reload enabled in development mode
- Full async support for high concurrency
- Integrated with MongoDB, Redis, and ChromaDB
- LangChain-powered AI features
- OCR processing capabilities

## Files Created

1. **run_backend.bat** - Windows server launcher
2. **run_backend.sh** - Linux/Mac server launcher
3. **setup_backend.bat** - Windows setup script
4. **setup_backend.sh** - Linux/Mac setup script
5. **BACKEND_SETUP.md** - Detailed setup documentation
6. **QUICK_START.md** - This file

## Environment Configuration

Key settings in `.env`:
- `HOST=0.0.0.0` - Listen on all interfaces
- `PORT=5001` - Default port
- `OPENAI_API_KEY` - OpenAI API key
- `MONGODB_URI` - MongoDB connection string
- `REDIS_URL` - Redis connection string

For more details, see [BACKEND_SETUP.md](BACKEND_SETUP.md)
