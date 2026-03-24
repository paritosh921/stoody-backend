import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { apiRequest, connectWs } from '../src/api-client';
import type { ApiError } from '../src/api-client';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function base64url(obj: Record<string, unknown>): string {
  const json = JSON.stringify(obj);
  const b64 = Buffer.from(json).toString('base64');
  return b64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fakeJwt(payload: Record<string, unknown>): string {
  const header = base64url({ alg: 'HS256', typ: 'JWT' });
  const body = base64url(payload);
  return `${header}.${body}.fake_sig`;
}

function mockResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 503 ? 'Service Unavailable' : 'Error',
    json: () => Promise.resolve(body),
    headers: new Headers(),
  } as unknown as Response;
}

// ---------------------------------------------------------------------------
// Setup / teardown
// ---------------------------------------------------------------------------

let fetchCalls: RequestInit[];

beforeEach(() => {
  fetchCalls = [];
  const store: Record<string, string> = {};
  vi.stubGlobal('localStorage', {
    getItem: (k: string) => store[k] ?? null,
    setItem: (k: string, v: string) => { store[k] = v; },
    removeItem: (k: string) => { delete store[k]; },
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// apiRequest adds auth headers from stored token
// ---------------------------------------------------------------------------

describe('apiRequest', () => {
  it('adds auth headers when a token is stored', async () => {
    const token = fakeJwt({ user_id: 'u1', exp: 9999999999 });
    localStorage.setItem('exampen_token', token);

    vi.stubGlobal('fetch', vi.fn((_url: string, init: RequestInit) => {
      fetchCalls.push(init);
      return Promise.resolve(mockResponse(200, { ok: true }));
    }));

    await apiRequest('GET', '/api/test', undefined, {
      baseUrl: 'http://localhost:8000',
    });

    expect(fetchCalls).toHaveLength(1);
    const headers = fetchCalls[0]!.headers as Record<string, string>;
    expect(headers['Authorization']).toBe(`Bearer ${token}`);
    expect(headers['X-Requested-With']).toBe('ExamPen');
  });

  // ---- Retry on 503 -------------------------------------------------------

  it('retries on 503 and succeeds on second attempt', async () => {
    let callCount = 0;
    vi.stubGlobal('fetch', vi.fn(() => {
      callCount++;
      if (callCount === 1) {
        return Promise.resolve(mockResponse(503, {}));
      }
      return Promise.resolve(mockResponse(200, { retried: true }));
    }));

    const result = await apiRequest<{ retried: boolean }>(
      'GET', '/api/health', { maxRetries: 2 },
      { baseUrl: 'http://localhost:8000' },
    );

    expect(callCount).toBe(2);
    expect(result.data.retried).toBe(true);
  });

  // ---- Throws ApiError on 4xx ----------------------------------------------

  it('throws ApiError on 4xx', async () => {
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve(mockResponse(404, {
        code: 'NOT_FOUND',
        message: 'Exam not found',
      })),
    ));

    try {
      await apiRequest('GET', '/api/exams/bad', undefined, {
        baseUrl: 'http://localhost:8000',
      });
      expect.unreachable('should have thrown');
    } catch (err) {
      const apiErr = err as ApiError;
      expect(apiErr.status).toBe(404);
      expect(apiErr.code).toBe('NOT_FOUND');
      expect(apiErr.message).toBe('Exam not found');
    }
  });
});

// ---------------------------------------------------------------------------
// connectWs constructs correct URL
// ---------------------------------------------------------------------------

describe('connectWs', () => {
  it('constructs correct WebSocket URL from baseUrl', () => {
    // Provide a minimal WebSocket mock that captures the URL
    let capturedUrl = '';
    vi.stubGlobal('WebSocket', class {
      onmessage: unknown = null;
      onerror: unknown = null;
      onclose: unknown = null;
      constructor(url: string) { capturedUrl = url; }
    });

    connectWs({
      onMessage: () => {},
      config: { baseUrl: 'http://localhost:8000' },
    });

    expect(capturedUrl).toBe('ws://localhost:8000/api/v1/invigilator/ws');
  });

  it('appends token as query parameter', () => {
    let capturedUrl = '';
    vi.stubGlobal('WebSocket', class {
      onmessage: unknown = null;
      onerror: unknown = null;
      onclose: unknown = null;
      constructor(url: string) { capturedUrl = url; }
    });

    connectWs({
      token: 'tok_123',
      onMessage: () => {},
      config: { baseUrl: 'https://api.example.com' },
    });

    expect(capturedUrl).toContain('wss://api.example.com');
    expect(capturedUrl).toContain('token=tok_123');
  });
});
