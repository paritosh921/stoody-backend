# SkillBot Backend Setup Guide

This guide will help you set up and run the SkillBot backend server.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

## Quick Start (Windows)

### First Time Setup

1. **Run the setup script** to create virtual environment and install dependencies:
   ```bash
   setup_backend.bat
   ```

2. **Configure environment variables**:
   - Ensure the `.env` file exists in the `backend` folder
   - Update API keys and configuration as needed

3. **Start the server**:
   ```bash
   run_backend.bat
   ```

### Subsequent Runs

Just run the server script:
```bash
run_backend.bat
```

## Quick Start (Linux/Mac)

### First Time Setup

1. **Make scripts executable**:
   ```bash
   chmod +x run_backend.sh
   chmod +x setup_backend.sh
   ```

2. **Run the setup script**:
   ```bash
   ./setup_backend.sh
   ```

3. **Configure environment variables**:
   - Ensure the `.env` file exists in the `backend` folder
   - Update API keys and configuration as needed

4. **Start the server**:
   ```bash
   ./run_backend.sh
   ```

### Subsequent Runs

Just run the server script:
```bash
./run_backend.sh
```

## Manual Setup

If you prefer to set up manually:

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment**:
   - Windows: `venv\Scripts\activate.bat`
   - Linux/Mac: `source venv/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**:
   ```bash
   python main_async.py
   ```

## Server Configuration

The server is configured via the `.env` file with the following key settings:

- **Host**: `0.0.0.0` (accessible from all network interfaces)
- **Port**: `5001` (default, can be changed in `.env`)
- **Mode**: Development (auto-reload enabled)

## Accessing the Server

Once the server is running, you can access:

- **API Documentation**: http://localhost:5001/docs
- **Health Check**: http://localhost:5001/health
- **Root Endpoint**: http://localhost:5001/

## Environment Variables

Key environment variables in `.env`:

```env
# Server
HOST=0.0.0.0
PORT=5001
NODE_ENV=development

# AI Provider
AI_PROVIDER=openai
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o

# Database
MONGODB_URI=your-mongodb-uri
MONGODB_DB_NAME=skillbot_db

# Authentication
JWT_SECRET_KEY=your-secret-key
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Redis Cache
REDIS_URL=redis://localhost:6379/0
```

## Troubleshooting

### Virtual Environment Not Found
If you see "Virtual environment not found", run the setup script first:
- Windows: `setup_backend.bat`
- Linux/Mac: `./setup_backend.sh`

### Port Already in Use
If port 5001 is already in use:
1. Change the `PORT` value in `.env`
2. Or stop the process using port 5001

### Dependency Installation Errors
If you encounter errors during installation:
1. Ensure Python 3.9+ is installed
2. Upgrade pip: `python -m pip install --upgrade pip`
3. Try installing dependencies individually

### MongoDB Connection Issues
- Ensure `MONGODB_URI` in `.env` is correct
- Check network connectivity to MongoDB server
- For development, you can set `DISABLE_MONGODB=true` in `.env`

### Redis Connection Issues
- Ensure Redis is running locally on port 6379
- Or update `REDIS_URL` in `.env` to point to your Redis instance
- In development mode, the app can run without Redis

## Development Mode

The server runs in development mode by default with:
- **Auto-reload**: Server restarts automatically when code changes
- **Detailed logging**: All requests and errors are logged
- **Debug endpoints**: `/docs` and `/redoc` are enabled

## Production Mode

For production deployment:
1. Set `NODE_ENV=production` in `.env`
2. Configure proper `FRONTEND_URL` for CORS
3. Set up Redis for caching and rate limiting
4. Use a process manager like PM2 or systemd
5. Set up proper logging and monitoring

## Architecture

The backend uses:
- **FastAPI**: Modern, high-performance web framework
- **Uvicorn**: ASGI server for async Python
- **MongoDB**: Primary database (via Motor async driver)
- **Redis**: Caching and rate limiting
- **ChromaDB**: Vector database for RAG/embeddings
- **LangChain**: AI/LLM orchestration

## API Endpoints

Main API routes:
- `/api/v1/chat` - Chat functionality
- `/api/v1/auth` - Authentication
- `/api/v1/admin` - Admin operations
- `/api/v1/student` - Student operations
- `/api/v1/tutor` - Tutor operations
- `/api/v1/questions` - Question management
- `/api/v1/images` - Image processing
- `/api/v1/practice` - Practice exercises
- `/api/v1/mcq` - Multiple choice questions
- `/api/v1/debugger` - Code debugging (optional)
- `/api/v1/learning` - Learning content
- `/api/v1/pdf` - PDF processing

## Support

For issues or questions:
1. Check the logs in `logs/app.log`
2. Review the API documentation at `/docs`
3. Check environment variable configuration
4. Ensure all required services (MongoDB, Redis) are running
