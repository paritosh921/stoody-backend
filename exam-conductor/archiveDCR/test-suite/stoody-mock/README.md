# Stoody Mock Server

Lightweight FastAPI application that simulates the Stoody education platform for ExamPen development and testing. Provides all Stoody API endpoints that ExamPen consumes, plus a JWKS endpoint for JWT validation.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/.well-known/jwks.json` | RSA public keys (JWKS format) |
| GET | `/api/users/{user_id}` | User profile (canned data) |
| GET | `/api/students?class_id=&section_id=` | Student roster |
| GET | `/api/tutors?subject_id=` | Tutor list |
| GET | `/api/classes` | Class list |
| GET | `/api/subjects` | Subject list |
| GET | `/api/parents/{user_id}/children` | Parent-child relationships |
| POST | `/api/webhooks/exampen/scores` | Score publication webhook (logs payload) |
| POST | `/api/webhooks/exampen/exams` | Exam lifecycle webhook (logs payload) |
| GET | `/debug/webhooks` | View received webhooks (dev only) |
| POST | `/debug/token?user_id=&role=&tenant_id=` | Generate signed test JWT (dev only) |

## Available Test Users

| user_id | role | name |
|---------|------|------|
| `tutor-001` | tutor | Rajesh Kumar |
| `tutor-002` | tutor | Priya Sharma |
| `student-001` | student | Arjun Mehta |
| `student-002` | student | Sneha Patel |
| `student-003` | student | Rohit Gupta |
| `parent-001` | parent | Vikram Mehta (children: student-001, student-003) |
| `admin-001` | admin | Dr. Sunita Reddy |

## Running Locally

```bash
cd test-suite/stoody-mock
pip install fastapi uvicorn pyjwt[crypto] cryptography
uvicorn main:app --port 9100 --reload
```

## Running with Docker

```bash
docker build -t stoody-mock .
docker run -p 9100:9100 stoody-mock
```

## Generating Test Tokens

```bash
# Via the /debug/token endpoint
curl -X POST "http://localhost:9100/debug/token?user_id=tutor-001&role=tutor"

# The returned JWT is signed with the mock RSA key and can be validated
# against the /.well-known/jwks.json endpoint.
```

## Architecture

- `main.py` — FastAPI app with all route handlers
- `data.py` — Canned response data (users, classes, subjects, etc.)
- `keys.py` — RSA keypair generation and JWT signing helpers

The RSA keypair is generated at process startup. The JWKS endpoint serves the public key. The `/debug/token` endpoint uses the private key to sign test JWTs. This allows `svc-auth` to validate tokens against this mock just as it would against a real Stoody server.
