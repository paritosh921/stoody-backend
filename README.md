# SkillBot Backend API

Flask backend server for the SkillBot learning platform with ChromaDB integration for persistent question storage and image management.

## Features

- **ChromaDB Integration**: Persistent storage for questions with semantic search capabilities
- **Image Management**: UUID-based file storage system for question images
- **RESTful API**: Complete CRUD operations for questions and images
- **CORS Support**: Configured for frontend integration
- **Error Handling**: Comprehensive error handling and validation

## Quick Start

### 1. Use Python 3.11

This backend must run on **Python 3.11**. Python 3.12 on Windows is known to trigger intermittent TLS handshake errors with MongoDB Atlas. Install Python 3.11 (alongside your existing interpreter if needed), then recreate the virtual environment with that interpreter:

```bash
# From the project root
cd backend
py -3.11 -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Start the Server

```bash
# Basic start
python run.py

# With debug mode
python run.py --debug

# Custom port
python run.py --port 5001
```

### 4. Test the API

```bash
# Health check
curl http://localhost:5000/health

# API documentation
curl http://localhost:5000/
```

## API Endpoints

### Questions API (`/api/questions`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/save` | Save a single question |
| POST | `/batch-save` | Save multiple questions |
| GET | `/<id>` | Get question by ID |
| GET | `/search` | Search questions with filters |
| PUT | `/<id>` | Update question |
| DELETE | `/<id>` | Delete question |
| GET | `/stats` | Get collection statistics |
| GET | `/export` | Export all questions |

### Images API (`/api/images`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload image file |
| POST | `/upload-base64` | Upload base64 image |
| GET | `/<path>` | Serve image file |
| GET | `/<path>/base64` | Get image as base64 |
| GET | `/<path>/info` | Get image information |
| DELETE | `/<path>` | Delete image |
| POST | `/cleanup` | Cleanup orphaned images |

## Data Storage

### ChromaDB
- **Location**: `./chromadb_data/`
- **Collection**: `questions`
- **Features**: Semantic search, metadata filtering, full-text search

### Images
- **Location**: `./images/`
- **Naming**: `{uuid}_{original_filename}`
- **Formats**: PNG, JPG, JPEG, GIF, BMP, WEBP
- **Max Size**: 10MB per image

## Usage Examples

### Save a Question

```python
import requests

question_data = {
    "id": "question_123",
    "text": "What is the capital of France?",
    "subject": "Geography",
    "difficulty": "easy",
    "options": ["London", "Berlin", "Paris", "Madrid"],
    "correctAnswer": "Paris",
    "images": [
        {
            "id": "img_1",
            "filename": "france_map.png",
            "description": "Map of France",
            "type": "diagram",
            "base64Data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
        }
    ]
}

response = requests.post(
    "http://localhost:5000/api/questions/save",
    json=question_data
)
print(response.json())
```

### Search Questions

```python
import requests

# Search by query
response = requests.get(
    "http://localhost:5000/api/questions/search",
    params={
        "query": "capital of France",
        "include_images": "true"
    }
)
questions = response.json()["questions"]

# Filter by subject and difficulty
response = requests.get(
    "http://localhost:5000/api/questions/search",
    params={
        "subject": "Geography",
        "difficulty": "easy",
        "limit": 10
    }
)
```

### Upload Image

```python
import requests

# Upload file
with open("image.png", "rb") as f:
    response = requests.post(
        "http://localhost:5000/api/images/upload",
        files={"file": f}
    )

# Upload base64
response = requests.post(
    "http://localhost:5000/api/images/upload-base64",
    json={
        "filename": "diagram.png",
        "base64Data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    }
)
```

## Configuration

Edit `config.py` to customize settings:

```python
# Database configuration
CHROMADB_COLLECTION_NAME = "questions"

# Image storage
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Flask configuration
FLASK_PORT = 5000
CORS_ORIGINS = ["http://localhost:8080", "http://127.0.0.1:8080"]
```

## Development

### Project Structure

```
backend/
├── app.py                  # Main Flask application
├── run.py                  # Server runner script
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── models/
│   ├── question.py        # Question data models
│   └── chromadb_client.py # ChromaDB client
├── services/
│   ├── question_service.py # Question business logic
│   └── image_service.py    # Image processing
├── routes/
│   ├── questions.py       # Question API routes
│   └── images.py          # Image API routes
├── chromadb_data/         # ChromaDB storage (auto-created)
└── images/                # Image files (auto-created)
```

### Adding New Features

1. **Add new model**: Create in `models/`
2. **Add business logic**: Create service in `services/`
3. **Add API routes**: Create blueprint in `routes/`
4. **Register blueprint**: Add to `app.py`

### Testing

```bash
# Install test dependencies
pip install pytest requests

# Run tests (when implemented)
pytest tests/
```

## Error Handling

The API returns consistent error responses:

```json
{
    "success": false,
    "error": "Error type",
    "message": "Detailed error message"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (validation error)
- `404`: Not Found
- `500`: Internal Server Error

## Logging

Logs are written to console with the format:
```
2024-01-01 12:00:00 - name - level - message
```

For production, configure file logging in `app.py`.

## Production Deployment

### Using Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:create_app()
```

### Environment Variables

```bash
export FLASK_DEBUG=false
export FLASK_PORT=5000
```

### Docker (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "run.py"]
```

## Troubleshooting

### Common Issues

1. **ChromaDB errors**: Ensure write permissions for `chromadb_data/`
2. **Image upload fails**: Check file size and format restrictions
3. **CORS errors**: Verify frontend URL in `config.py`
4. **Port conflicts**: Use `--port` flag to change port

### Debug Mode

Run with debug mode for detailed error information:

```bash
python run.py --debug
```

## License

Part of the SkillBot Learning Platform project.
