/**
 * Auth hook for the student portal.
 *
 * Follows the Stoody embed pattern: Stoody SPA passes the JWT via URL query
 * param (`?token=...`) on first load.  The hook stores it in localStorage
 * under `exampen_token` and subsequent requests use it from there.
 *
 * Provides login(token), logout(), and decoded user info.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  parseJwtClaims,
  isTokenExpired,
  storeToken,
  getToken,
  clearToken,
  type ExamPenClaims,
} from "@exampen/common-ts";

export interface AuthUser {
  user_id: string;
  name: string;
  role: "student" | "parent";
}

export interface UseAuthResult {
  user: AuthUser | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (token: string) => void;
  logout: () => void;
}

function decodeUser(token: string): AuthUser | null {
  try {
    const claims: ExamPenClaims = parseJwtClaims(token);
    const role =
      claims.stoody_role === "parent" ? "parent" : "student";
    return {
      user_id: claims.user_id,
      name: claims.name,
      role: role as "student" | "parent",
    };
  } catch {
    return null;
  }
}

export function useAuth(): UseAuthResult {
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
    const urlToken = params.get("token");
    if (urlToken && !isTokenExpired(urlToken)) {
      storeToken(urlToken);
      setToken(urlToken);
      // Remove the token from the URL to avoid leaking it in browser history
      const url = new URL(window.location.href);
      url.searchParams.delete("token");
      window.history.replaceState({}, "", url.toString());
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

  const user = useMemo(() => {
    if (!token) return null;
    return decodeUser(token);
  }, [token]);

  const isAuthenticated = token !== null && !isTokenExpired(token);

  return { user, token, isAuthenticated, login, logout };
}
