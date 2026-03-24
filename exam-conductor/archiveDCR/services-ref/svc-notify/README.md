# svc-notify

Notification service for ExamPen. Subscribes to NATS events (`score.updated`,
`objection.*`, `exam.lifecycle`) and dispatches notifications via email, push,
and SMS channels.

## Architecture

```
events/event_consumers.py   ← NATS subscriptions
        ↓
domain/trigger_rules.py     ← pure: event → list[NotificationAction]
domain/templates.py         ← pure: action → rendered content
        ↓
adapters/dispatcher.py      ← route to channel adapter with retry
adapters/email_sender.py    ← aiosmtplib
adapters/push_sender.py     ← FCM stub
adapters/sms_sender.py      ← Twilio stub
```

The `domain/` layer performs **zero I/O** — it maps events to notification
actions and renders templates as pure functions.

## Running

```bash
# Local dev
uvicorn src.main:app --reload --port 8007

# Docker
docker build -t svc-notify .
docker run -e NATS_URL=nats://nats:4222 -p 8007:8007 svc-notify

# Tests
pytest tests/ -m unit
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NATS_URL` | `nats://localhost:4222` | NATS server URL |
| `SMTP_HOST` | `localhost` | SMTP server host |
| `SMTP_PORT` | `587` | SMTP server port |
| `SMTP_USERNAME` | (empty) | SMTP auth username |
| `SMTP_PASSWORD` | (empty) | SMTP auth password |
| `SMTP_FROM_EMAIL` | `noreply@exampen.local` | Sender email address |
| `PUSH_FCM_SERVER_KEY` | (empty) | FCM server key for push |
| `PUSH_FCM_API_URL` | `https://fcm.googleapis.com/fcm/send` | FCM endpoint |
| `SMS_TWILIO_ACCOUNT_SID` | (empty) | Twilio account SID |
| `SMS_TWILIO_AUTH_TOKEN` | (empty) | Twilio auth token |
| `SMS_TWILIO_FROM_NUMBER` | (empty) | Twilio sender number |
