# B2C User Support Documentation

## Overview

The B2C flow allows individual students to purchase and use the Stoody platform directly without going through an institute (B2B/CRM-based) system.

## B2C vs B2B Comparison

| Feature | B2B (CRM) | B2C (Direct) |
|---------|-----------|--------------|
| Account Creation | Admin creates student accounts | Self-registration via Google OAuth |
| Database | `skillbot_db` | `STOODY-b2c` |
| Authentication | Username/Password (assigned by admin) | Google OAuth |
| Plan Selection | Assigned by admin | Student selects during onboarding |
| Content Management | Institute admin | B2C Admin |

## B2C User Flow

### 1. Sign Up / Login
- Students visit the login page and click "Sign in with Google"
- After Google OAuth, they are redirected to the platform
- First-time users are redirected to onboarding

### 2. Onboarding (First-time Users)
- **Step 1: Plan Selection**
  - Select exam type: JEE or NEET
  - Select class level: 9, 10, 11, 12, or Dropper
  
- **Step 2: Personal Details**
  - Full name (required)
  - Phone number (required)
  - School/College name (optional)
  - City (optional)

### 3. Dashboard Access
- After completing onboarding, students can access:
  - Dashboard
  - Learning mode
  - MCQ practice
  - All content relevant to their selected plan

## Available Plans

### JEE (Engineering)
- Classes: 9, 10, 11, 12, Dropper
- Subjects: Physics, Chemistry, Mathematics

### NEET (Medical)
- Classes: 9, 10, 11, 12, Dropper
- Subjects: Physics, Chemistry, Biology

## API Endpoints

### B2C Authentication (`/api/v1/b2c/`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/google/login` | POST | Google OAuth login/signup |
| `/me` | GET | Get current B2C user |
| `/verify` | GET | Verify JWT token |
| `/logout` | POST | Logout B2C user |

### B2C Profile & Onboarding (`/api/v1/b2c/`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/profile/onboarding` | POST | Complete onboarding with plan selection |
| `/profile` | GET | Get full B2C user profile |
| `/profile` | PUT | Update B2C user profile |
| `/profile/check-onboarding` | GET | Check if onboarding is complete |

### B2C Admin (`/api/v1/b2c/admin/`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/login` | POST | B2C Admin login |
| `/setup` | POST | Initial admin setup |
| `/me` | GET | Get B2C admin profile |
| `/dashboard/stats` | GET | Dashboard statistics |
| `/students` | GET | List B2C students |

## Database Schema

### B2C Users Collection (`STOODY-b2c.users`)

```javascript
{
  _id: ObjectId,
  google_id: String,
  email: String,
  full_name: String,
  given_name: String,
  family_name: String,
  picture: String,
  phone: String,
  school_name: String,
  city: String,
  locale: String,
  
  // User status
  is_active: Boolean,
  user_type: "b2c_user",
  
  // Plan & Learning
  exam_type: "JEE" | "NEET",
  class_level: "9" | "10" | "11" | "12" | "Dropper",
  standard: String,
  subjects: ["Physics", "Chemistry", "Mathematics"] | ["Physics", "Chemistry", "Biology"],
  plan_types: ["JEE"] | ["NEET"],
  is_dropper: Boolean,
  
  // Onboarding
  onboarding_complete: Boolean,
  onboarding_completed_at: Date,
  
  // Timestamps
  created_at: Date,
  updated_at: Date,
  last_login: Date
}
```

## Environment Variables

Add to `.env`:

```env
# B2C Database
MONGODB_DB_STOODY=STOODY-b2c

# Google OAuth (for B2C users)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# B2C Admin Setup
B2C_ADMIN_SETUP_KEY=your_secure_setup_key
```

## Content Access Logic

B2C students can only see content that:
1. Is uploaded by the **B2C Admin**
2. Matches their **exam type** (JEE or NEET)
3. Matches their **class level** (or is marked for all levels)

Content filtering is done using:
- `plan_types` field (matches exam_type)
- `standard` field (matches class_level)

## Setting Up B2C Admin

1. Ensure the backend is running with the B2C database connected
2. Make a POST request to `/api/v1/b2c/admin/setup`:

```json
{
  "username": "b2cadmin",
  "password": "securepassword",
  "email": "admin@example.com",
  "full_name": "B2C Administrator",
  "setup_key": "stoody-b2c-admin-setup-2024"
}
```

3. Only one B2C Admin account is allowed

## Frontend Routes

| Route | Component | Description |
|-------|-----------|-------------|
| `/login` | Login | Combined login (includes Google OAuth) |
| `/onboarding` | B2COnboarding | Plan selection & details |
| `/profile` | B2CProfile | B2C user profile |
| `/dashboard` | Dashboard | Student dashboard |
| `/b2c-admin-login` | B2CAdminLogin | B2C Admin login |
| `/b2c-admin` | B2CAdminDashboard | B2C Admin panel |

## Future Enhancements (Not Implemented)

1. **Payment Gateway Integration**
   - Razorpay/Stripe integration for plan purchases
   - Subscription management
   
2. **Plan-Based Content Locking**
   - Free vs Premium content
   - Trial periods
   
3. **B2C Analytics**
   - User engagement tracking
   - Plan conversion metrics
