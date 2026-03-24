"""Environment configuration for svc-notify."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All settings loaded from environment variables with sensible defaults."""

    model_config = {"env_prefix": ""}

    # NATS
    nats_url: str = "nats://localhost:4222"

    # SMTP (email)
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    smtp_from_email: str = "noreply@exampen.local"

    # Push (FCM)
    push_fcm_server_key: str = ""
    push_fcm_api_url: str = "https://fcm.googleapis.com/fcm/send"

    # SMS (Twilio)
    sms_twilio_account_sid: str = ""
    sms_twilio_auth_token: str = ""
    sms_twilio_from_number: str = ""

    # Dispatcher
    dispatch_max_retries: int = 3
    dispatch_base_delay_s: float = 1.0

    # Stoody webhooks
    stoody_webhook_url: str = "http://localhost:9100"
    stoody_webhook_secret: str = ""
    stoody_webhook_max_retries: int = 3
    stoody_webhook_base_delay_s: float = 2.0


settings = Settings()
