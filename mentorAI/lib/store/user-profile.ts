/**
 * User Profile Store
 * Persists avatar, nickname & bio to localStorage
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { withBasePath } from '@/lib/utils/base-path';

/** Predefined avatar options */
export const AVATAR_OPTIONS = [
  withBasePath('/avatars/user.png'),
  withBasePath('/avatars/teacher-2.png'),
  withBasePath('/avatars/assist-2.png'),
  withBasePath('/avatars/clown-2.png'),
  withBasePath('/avatars/curious-2.png'),
  withBasePath('/avatars/note-taker-2.png'),
  withBasePath('/avatars/thinker-2.png'),
] as const;

export interface UserProfileState {
  /** Local avatar path or data-URL (for custom uploads) */
  avatar: string;
  nickname: string;
  bio: string;
  setAvatar: (avatar: string) => void;
  setNickname: (nickname: string) => void;
  setBio: (bio: string) => void;
}

export const useUserProfileStore = create<UserProfileState>()(
  persist(
    (set) => ({
      avatar: AVATAR_OPTIONS[0],
      nickname: '',
      bio: '',
      setAvatar: (avatar) => set({ avatar }),
      setNickname: (nickname) => set({ nickname }),
      setBio: (bio) => set({ bio }),
    }),
    {
      name: 'user-profile-storage',
    },
  ),
);
