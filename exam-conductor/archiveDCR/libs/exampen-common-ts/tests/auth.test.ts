import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  parseJwtClaims,
  isTokenExpired,
  getAuthHeaders,
  storeToken,
  getToken,
  clearToken,
} from '../src/auth';

// ---------------------------------------------------------------------------
// Helpers: build fake JWTs from plain objects
// ---------------------------------------------------------------------------

function base64url(obj: Record<string, unknown>): string {
  const json = JSON.stringify(obj);
  const b64 = Buffer.from(json).toString('base64');
  return b64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fakeJwt(payload: Record<string, unknown>): string {
  const header = base64url({ alg: 'HS256', typ: 'JWT' });
  const body = base64url(payload);
  const sig = 'fake_signature';
  return `${header}.${body}.${sig}`;
}

// ---------------------------------------------------------------------------
// parseJwtClaims
// ---------------------------------------------------------------------------

describe('parseJwtClaims', () => {
  it('decodes a valid JWT payload correctly', () => {
    const payload = {
      user_id: 'u_001',
      tenant_id: 't_001',
      stoody_role: 'tutor',
      exampen_roles: ['tutor', 'evaluator'],
      name: 'Dr. Anita Sharma',
      email: 'anita@school.example.com',
      exp: 1999999999,
      iat: 1700000000,
    };
    const token = fakeJwt(payload);
    const claims = parseJwtClaims(token);

    expect(claims.user_id).toBe('u_001');
    expect(claims.tenant_id).toBe('t_001');
    expect(claims.stoody_role).toBe('tutor');
    expect(claims.exampen_roles).toEqual(['tutor', 'evaluator']);
    expect(claims.name).toBe('Dr. Anita Sharma');
    expect(claims.email).toBe('anita@school.example.com');
    expect(claims.exp).toBe(1999999999);
    expect(claims.iat).toBe(1700000000);
  });

  it('throws on a malformed JWT (wrong number of parts)', () => {
    expect(() => parseJwtClaims('only.two')).toThrow('Malformed JWT');
    expect(() => parseJwtClaims('')).toThrow('Malformed JWT');
  });
});

// ---------------------------------------------------------------------------
// isTokenExpired
// ---------------------------------------------------------------------------

describe('isTokenExpired', () => {
  it('returns true for an expired token', () => {
    const past = Math.floor(Date.now() / 1000) - 3600; // 1 hour ago
    const token = fakeJwt({ exp: past });
    expect(isTokenExpired(token)).toBe(true);
  });

  it('returns false for a token with future expiry', () => {
    const future = Math.floor(Date.now() / 1000) + 3600; // 1 hour from now
    const token = fakeJwt({ exp: future });
    expect(isTokenExpired(token)).toBe(false);
  });

  it('returns true for a malformed token', () => {
    expect(isTokenExpired('garbage')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// getAuthHeaders
// ---------------------------------------------------------------------------

describe('getAuthHeaders', () => {
  it('returns correct Authorization and CSRF headers', () => {
    const headers = getAuthHeaders('my_token_123');
    expect(headers['Authorization']).toBe('Bearer my_token_123');
    expect(headers['X-Requested-With']).toBe('ExamPen');
  });
});

// ---------------------------------------------------------------------------
// Token storage round-trip (mock localStorage)
// ---------------------------------------------------------------------------

describe('token storage', () => {
  const store: Record<string, string> = {};

  beforeEach(() => {
    // Mock localStorage on globalThis
    const mockStorage = {
      getItem: vi.fn((key: string) => store[key] ?? null),
      setItem: vi.fn((key: string, val: string) => { store[key] = val; }),
      removeItem: vi.fn((key: string) => { delete store[key]; }),
    };
    vi.stubGlobal('localStorage', mockStorage);
  });

  afterEach(() => {
    for (const key of Object.keys(store)) delete store[key];
    vi.unstubAllGlobals();
  });

  it('store -> get -> clear -> get returns null', () => {
    const token = fakeJwt({ user_id: 'u_roundtrip', exp: 9999999999 });

    storeToken(token);
    expect(getToken()).toBe(token);

    clearToken();
    expect(getToken()).toBeNull();
  });
});
