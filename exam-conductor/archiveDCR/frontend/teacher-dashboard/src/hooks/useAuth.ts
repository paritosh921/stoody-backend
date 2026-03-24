// ---------------------------------------------------------------------------
// Auth hook for the teacher dashboard.
//
// Follows the Stoody embed pattern: Stoody SPA passes the JWT via URL query
// param (`?token=...`) on first load.  The hook stores it in localStorage
// under `exampen_token` and subsequent requests use it from there.
//
// The shared common-ts apiRequest() already reads the token from localStorage
// via getToken(), so this hook mainly provides reactive auth state for the UI.
// ---------------------------------------------------------------------------

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  parseJwtClaims,
  isTokenExpired,
  storeToken,
  getToken,
  clearToken,
  type ExamPenClaims,
} from '@exampen/common-ts';

export interface AuthContext {
  token: string | null;
  userId: string;
  displayName: string;
  isAuthenticated: boolean;
  login: (token: string) => void;
  logout: () => void;
}

function decodeClaims(token: string): ExamPenClaims | null {
  try {
    return parseJwtClaims(token);
  } catch {
    return null;
  }
}

export function useAuth(): AuthContext {
  const [token, setToken] = useState<string | null>(() => {
    const stored = getToken();
    if (stored && !isTokenExpired(stored)) return stored;
    // Clear expired tokens
    if (stored) clearToken();
    return null;
  });

  // On first mount, check for token in URL query params (Stoody embed pattern)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const urlToken = params.get('token');
    if (urlToken && !isTokenExpired(urlToken)) {
      storeToken(urlToken);
      setToken(urlToken);
      // Remove the token from the URL to avoid leaking it in browser history
      const url = new URL(window.location.href);
      url.searchParams.delete('token');
      window.history.replaceState({}, '', url.toString());
    }
  }, []);

  const login = useCallback((newToken: string) => {
    storeToken(newToken);
    setToken(newToken);
  }, []);

  const logout = useCallback(() => {
    clearToken();
    setToken(null);
  }, []);

  const claims = useMemo(() => {
    if (!token) return null;
    return decodeClaims(token);
  }, [token]);

  const isAuthenticated = token !== null && !isTokenExpired(token);

  return {
    token,
    userId: claims?.user_id ?? '',
    displayName: claims?.name ?? '',
    isAuthenticated,
    login,
    logout,
  };
}
