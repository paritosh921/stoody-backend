# B2C User Support Documentation

## Overview

This document describes the B2C (Business-to-Consumer) user support added to the Stoody platform. B2C users are individual users who sign in using Google OAuth, separate from the existing school/institution-based users.

## Architecture

### Database Separation

The system now supports two completely isolated databases:

| Database | Purpose | Users |
|----------|---------|-------|
| `skillbot_db` | Original database | Admins, Teachers, Students (school-based) |
| `stoody-b2c` | B2C database | Individual B2C users (Google OAuth) |

**Important**: These databases are completely isolated. No data is shared between them.

### Authentication Flows

#### Existing Users (Unchanged)
- **Admin**: Email + Password → `skillbot_db.admins`
- **Teacher**: Username + Password → `skillbot_db.tutors`
- **Student**: Username + Password → `skillbot_db.students`

#### B2C Users (New)
- **B2C User**: Google Sign-In → `stoody-b2c.users`

## API Endpoints

### B2C Authentication Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/b2c/google/login` | POST | Google OAuth login/signup |
| `/api/v1/b2c/me` | GET | Get B2C user profile |
| `/api/v1/b2c/logout` | POST | B2C user logout |
| `/api/v1/b2c/verify` | GET | Verify B2C JWT token |

### Request/Response Examples

#### Google Login

**Request:**
```json
POST /api/v1/b2c/google/login
{
  "credential": "<Google OAuth ID Token>"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "<JWT Token>",
    "user_type": "b2c_user",
    "user": {
      "user_id": "...",
      "email": "user@gmail.com",
      "full_name": "John Doe",
      "picture": "https://...",
      "is_b2c": true
    }
  }
}
```

## Environment Configuration

### Backend (s-backend/.env)

Add the following environment variables:

```env
# B2C Database
MONGODB_DB_STOODY=stoody-b2c

# Google OAuth for B2C
GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET=YOUR_GOOGLE_CLIENT_SECRET
```

### Frontend (skiller-bot/.env.local)

```env
VITE_GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID
```

## User Types

| User Type | Database | Authentication | JWT `user_type` |
|-----------|----------|----------------|-----------------|
| Admin | skillbot_db | Email/Password | `admin` |
| Teacher | skillbot_db | Username/Password | `tutor` |
| Student | skillbot_db | Username/Password | `student` |
| B2C User | stoody-b2c | Google OAuth | `b2c_user` |

## Frontend Integration

### Google Sign-In Button

The Login page now includes a Google Sign-In button for B2C users:

```tsx
import { GoogleLogin, CredentialResponse } from '@react-oauth/google';

// In the Login component
<GoogleLogin
  onSuccess={handleGoogleSuccess}
  onError={handleGoogleError}
  useOneTap
  theme="outline"
  size="large"
/>
```

### User Detection

To detect if a user is a B2C user:

```tsx
const { user } = useAuth();

if (user?.isB2C || user?.userType === 'b2c_user') {
  // B2C user logic
}
```

## Database Collections

### stoody-b2c Database Collections

| Collection | Description |
|------------|-------------|
| `users` | B2C user profiles |
| `user_activity_log` | B2C user activity tracking |

### User Document Schema

```json
{
  "_id": ObjectId,
  "google_id": "Google OAuth sub ID",
  "email": "user@gmail.com",
  "full_name": "John Doe",
  "given_name": "John",
  "family_name": "Doe",
  "picture": "https://...",
  "locale": "en",
  "is_active": true,
  "user_type": "b2c_user",
  "created_at": ISODate,
  "last_login": ISODate,
  "admin_id": null,
  "subdomain": null
}
```

## Security Notes

1. **Token Verification**: Google OAuth tokens are verified using the `google-auth` library
2. **Database Isolation**: B2C users cannot access data from skillbot_db and vice versa
3. **JWT Security**: Same JWT mechanism is used for all user types
4. **Rate Limiting**: Standard rate limits apply to B2C endpoints

## Files Modified/Created

### Backend
- `config_async.py` - Added B2C database and Google OAuth config
- `core/database.py` - Added B2C database methods (b2c_find, b2c_insert, etc.)
- `api/v1/b2c_auth_async.py` - **NEW** - B2C authentication routes
- `main_async.py` - Added B2C router registration
- `requirements.txt` - Added google-auth dependency

### Frontend
- `src/App.tsx` - Added GoogleOAuthProvider wrapper
- `src/contexts/AuthContext.tsx` - Added loginWithGoogle implementation
- `src/pages/Login.tsx` - Added Google Sign-In button
- `.env.example` - Added Google Client ID

## Testing

1. Start the backend: `python main_async.py`
2. Start the frontend: `npm run dev`
3. Navigate to `/login`
4. Click "Sign in with Google" button
5. Complete Google OAuth flow
6. User should be redirected to dashboard

## Troubleshooting

### "Google token verification failed"
- Ensure `GOOGLE_CLIENT_ID` matches in both frontend and backend
- Check that the Google OAuth app is configured correctly

### "B2C MongoDB not connected"
- Ensure `MONGODB_URI` is set and accessible
- Check that the MongoDB user has access to `stoody-b2c` database

### Google Sign-In button not showing
- Ensure `VITE_GOOGLE_CLIENT_ID` is set in frontend `.env.local`
- Clear browser cache and restart dev server
