/** Chat types: messages, threads, read receipts. */

/** A single chat message in an exam thread. */
export interface ChatMessage {
  message_id: string;
  sender_id: string;
  recipient_id: string;
  exam_id: string;
  content: string;
  attachment_uri?: string;
  sent_at: string;
  read_at?: string;
}

/** Request to append a message to a chat thread. */
export interface SendChatMessageRequest {
  content: string;
  attachment_uri?: string;
}

/** Read receipt for a chat thread. */
export interface ReadReceipt {
  exam_id: string;
  other_user_id: string;
  read_at: string;
}

/** Generic message model used by BFF surfaces. */
export interface Message {
  message_id: string;
  sender_id: string;
  content: string;
  attachment_uri?: string;
  sent_at: string;
  read_at?: string;
}

/** Request to send a message (BFF surface). */
export interface SendMessageRequest {
  content: string;
  attachment_uri?: string;
}
