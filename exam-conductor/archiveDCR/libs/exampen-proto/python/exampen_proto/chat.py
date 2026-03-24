"""Chat models: messages, threads, read receipts."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """A single chat message in an exam thread."""

    message_id: UUID
    sender_id: str
    recipient_id: str
    exam_id: UUID
    content: str
    attachment_uri: Optional[str] = None
    sent_at: datetime
    read_at: Optional[datetime] = None


class SendChatMessageRequest(BaseModel):
    """Request to append a message to a chat thread."""

    content: str
    attachment_uri: Optional[str] = None


class ReadReceipt(BaseModel):
    """Read receipt for a chat thread."""

    exam_id: UUID
    other_user_id: str
    read_at: datetime


class Message(BaseModel):
    """Generic message model used by BFF surfaces."""

    message_id: UUID
    sender_id: str
    content: str
    attachment_uri: Optional[str] = None
    sent_at: datetime
    read_at: Optional[datetime] = None


class SendMessageRequest(BaseModel):
    """Request to send a message (BFF surface)."""

    content: str
    attachment_uri: Optional[str] = None
