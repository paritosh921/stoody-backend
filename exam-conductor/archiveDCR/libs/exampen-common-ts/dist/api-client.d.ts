import type { WebSocketEnvelope } from './types-hub';
export interface ApiResponse<T> {
    data: T;
    status: number;
}
export interface ApiError {
    code: string;
    message: string;
    status: number;
}
export declare function isApiError(value: unknown): value is ApiError;
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
export interface RequestOptions {
    body?: unknown;
    headers?: Record<string, string>;
    params?: Record<string, string>;
    signal?: AbortSignal;
    /** Override the default max-retry count for 503s (default: 2). */
    maxRetries?: number;
}
export type RequestInterceptor = (url: string, init: RequestInit) => RequestInit | Promise<RequestInit>;
export type ResponseInterceptor = (response: Response) => Response | Promise<Response>;
export interface ApiClientConfig {
    baseUrl?: string;
    defaultHeaders?: Record<string, string>;
    requestInterceptors?: RequestInterceptor[];
    responseInterceptors?: ResponseInterceptor[];
}
export declare function apiRequest<T>(method: HttpMethod, path: string, options?: RequestOptions, config?: ApiClientConfig): Promise<ApiResponse<T>>;
export declare function apiGet<T>(path: string, options?: Omit<RequestOptions, 'body'>, config?: ApiClientConfig): Promise<ApiResponse<T>>;
export declare function apiPost<T>(path: string, body: unknown, options?: RequestOptions, config?: ApiClientConfig): Promise<ApiResponse<T>>;
export declare function apiPatch<T>(path: string, body: unknown, options?: RequestOptions, config?: ApiClientConfig): Promise<ApiResponse<T>>;
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
export declare function connectWs(opts: WsClientOptions): WebSocket;
//# sourceMappingURL=api-client.d.ts.map