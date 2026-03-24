// ---------------------------------------------------------------------------
// Typed API client with auth injection, retry on 503, and interceptors.
// ---------------------------------------------------------------------------

import { getAuthHeaders, getToken } from './auth';
import type { WebSocketEnvelope } from './types-hub';

// ---- Response / Error types -----------------------------------------------

export interface ApiResponse<T> {
  data: T;
  status: number;
}

export interface ApiError {
  code: string;
  message: string;
  status: number;
}

export function isApiError(value: unknown): value is ApiError {
  return (
    typeof value === 'object' &&
    value !== null &&
    'code' in value &&
    'message' in value &&
    'status' in value
  );
}

// ---- Request options ------------------------------------------------------

export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';

export interface RequestOptions {
  body?: unknown;
  headers?: Record<string, string>;
  params?: Record<string, string>;
  signal?: AbortSignal;
  /** Override the default max-retry count for 503s (default: 2). */
  maxRetries?: number;
}

// ---- Interceptors ---------------------------------------------------------

export type RequestInterceptor = (
  url: string,
  init: RequestInit,
) => RequestInit | Promise<RequestInit>;

export type ResponseInterceptor = (
  response: Response,
) => Response | Promise<Response>;

// ---- Client configuration -------------------------------------------------

export interface ApiClientConfig {
  baseUrl?: string;
  defaultHeaders?: Record<string, string>;
  requestInterceptors?: RequestInterceptor[];
  responseInterceptors?: ResponseInterceptor[];
}

// ---- Resolve base URL -----------------------------------------------------

function resolveBaseUrl(override?: string): string {
  if (override) return override.replace(/\/+$/, '');

  // Vite projects — access import.meta.env safely
  try {
    const meta = import.meta as unknown as Record<string, unknown>;
    const env = meta['env'] as Record<string, string> | undefined;
    const viteUrl = env?.['VITE_EXAMPEN_API_URL'];
    if (viteUrl) return viteUrl.replace(/\/+$/, '');
  } catch {
    // import.meta.env unavailable outside Vite
  }

  // Node / generic — access process.env safely
  try {
    const g = globalThis as Record<string, unknown>;
    const proc = g['process'] as { env: Record<string, string | undefined> } | undefined;
    const nodeUrl = proc?.env['EXAMPEN_API_URL'];
    if (nodeUrl) return nodeUrl.replace(/\/+$/, '');
  } catch {
    // process unavailable in browser
  }

  return '';
}

// ---- Retry helper ---------------------------------------------------------

const DEFAULT_MAX_RETRIES = 2;
const RETRY_DELAY_MS = 500;

async function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---- API request ----------------------------------------------------------

export async function apiRequest<T>(
  method: HttpMethod,
  path: string,
  options?: RequestOptions,
  config?: ApiClientConfig,
): Promise<ApiResponse<T>> {
  const baseUrl = resolveBaseUrl(config?.baseUrl);
  const maxRetries = options?.maxRetries ?? DEFAULT_MAX_RETRIES;

  // Build URL with query params
  let url = `${baseUrl}${path.startsWith('/') ? path : `/${path}`}`;
  if (options?.params) {
    const qs = new URLSearchParams(options.params).toString();
    url = `${url}?${qs}`;
  }

  // Build headers
  const token = getToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(config?.defaultHeaders ?? {}),
    ...(token ? getAuthHeaders(token) : {}),
    ...(options?.headers ?? {}),
  };

  let init: RequestInit = {
    method,
    headers,
    body: options?.body !== undefined ? JSON.stringify(options.body) : undefined,
    signal: options?.signal,
  };

  // Apply request interceptors
  if (config?.requestInterceptors) {
    for (const interceptor of config.requestInterceptors) {
      init = await interceptor(url, init);
    }
  }

  // Execute with retry on 503
  let lastError: ApiError | undefined;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    let response = await fetch(url, init);

    // Apply response interceptors
    if (config?.responseInterceptors) {
      for (const interceptor of config.responseInterceptors) {
        response = await interceptor(response);
      }
    }

    if (response.status === 503 && attempt < maxRetries) {
      await delay(RETRY_DELAY_MS * (attempt + 1));
      continue;
    }

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({
        code: 'UNKNOWN',
        message: response.statusText,
      }));
      lastError = {
        code: (errorBody as Record<string, string>).code ?? 'UNKNOWN',
        message:
          (errorBody as Record<string, string>).message ?? response.statusText,
        status: response.status,
      };
      throw lastError;
    }

    const data = (await response.json()) as T;
    return { data, status: response.status };
  }

  // Should not reach here, but satisfies the type checker
  throw lastError ?? { code: 'RETRY_EXHAUSTED', message: '503 retries exhausted', status: 503 };
}

// ---- Convenience methods --------------------------------------------------

export function apiGet<T>(
  path: string,
  options?: Omit<RequestOptions, 'body'>,
  config?: ApiClientConfig,
): Promise<ApiResponse<T>> {
  return apiRequest<T>('GET', path, options, config);
}

export function apiPost<T>(
  path: string,
  body: unknown,
  options?: RequestOptions,
  config?: ApiClientConfig,
): Promise<ApiResponse<T>> {
  return apiRequest<T>('POST', path, { ...options, body }, config);
}

export function apiPatch<T>(
  path: string,
  body: unknown,
  options?: RequestOptions,
  config?: ApiClientConfig,
): Promise<ApiResponse<T>> {
  return apiRequest<T>('PATCH', path, { ...options, body }, config);
}

// ---- WebSocket helper for invigilator console -----------------------------

export interface WsClientOptions {
  /** Full ws:// or wss:// URL. Falls back to baseUrl + path. */
  url?: string;
  path?: string;
  token?: string;
  onMessage: (envelope: WebSocketEnvelope) => void;
  onError?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  config?: ApiClientConfig;
}

/**
 * Open a WebSocket connection for live invigilator feeds.
 * Returns the WebSocket instance so the caller can close it.
 */
export function connectWs(opts: WsClientOptions): WebSocket {
  const baseUrl = resolveBaseUrl(opts.config?.baseUrl);
  const wsBase = baseUrl.replace(/^http/, 'ws');
  const wsUrl = opts.url ?? `${wsBase}${opts.path ?? '/api/v1/invigilator/ws'}`;

  const tokenVal = opts.token ?? getToken();
  const fullUrl = tokenVal ? `${wsUrl}?token=${encodeURIComponent(tokenVal)}` : wsUrl;

  const ws = new WebSocket(fullUrl);

  ws.onmessage = (event: MessageEvent) => {
    const envelope = JSON.parse(String(event.data)) as WebSocketEnvelope;
    opts.onMessage(envelope);
  };

  if (opts.onError) ws.onerror = opts.onError;
  if (opts.onClose) ws.onclose = opts.onClose;

  return ws;
}
