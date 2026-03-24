// ---------------------------------------------------------------------------
// API client wrapping @exampen/common-ts apiRequest with auth injection.
// Base URL sourced from VITE_EXAMPEN_API_URL (defaults to '' for proxy).
// ---------------------------------------------------------------------------

import {
  apiRequest,
  type ApiClientConfig,
  type HttpMethod,
  type RequestOptions,
  type ApiResponse,
} from '@exampen/common-ts';

const config: ApiClientConfig = {
  baseUrl: import.meta.env.VITE_EXAMPEN_API_URL ?? '',
};

/**
 * Typed API caller that delegates to the shared `apiRequest`.
 * The common-ts client automatically reads the token from localStorage
 * via `getToken()` and injects Authorization + CSRF headers.
 */
export function request<T>(
  method: HttpMethod,
  path: string,
  options?: RequestOptions,
): Promise<ApiResponse<T>> {
  return apiRequest<T>(method, path, options, config);
}

export function get<T>(
  path: string,
  params?: Record<string, string>,
): Promise<ApiResponse<T>> {
  return request<T>('GET', path, params ? { params } : undefined);
}

export function post<T>(
  path: string,
  body: unknown,
): Promise<ApiResponse<T>> {
  return request<T>('POST', path, { body });
}

export function patch<T>(
  path: string,
  body: unknown,
): Promise<ApiResponse<T>> {
  return request<T>('PATCH', path, { body });
}
