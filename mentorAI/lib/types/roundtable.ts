import { withBasePath } from '@/lib/utils/base-path';

export const USER_AVATAR = withBasePath('/avatars/user.png');

export type ParticipantRole = 'teacher' | 'student' | 'user';

export interface Participant {
  id: string;
  name: string;
  role: ParticipantRole;
  avatar: string;
  isOnline: boolean;
  isSpeaking?: boolean;
}

export interface MessageAction {
  id: string;
  label: string;
  icon?: string;
  onClick: () => void;
}

export interface Message {
  id: string;
  senderId: string;
  senderRole: ParticipantRole;
  content: string;
  timestamp: number;
  actions?: MessageAction[];
}
