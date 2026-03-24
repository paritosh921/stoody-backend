// ---------------------------------------------------------------------------
// Typed API client with auth injection, retry on 503, and interceptors.
// ---------------------------------------------------------------------------
import { getAuthHeaders, getToken } from './auth';
export function isApiError(value) {
    return (typeof value === 'object' &&
        value !== null &&
        'code' in value &&
        'message' in value &&
        'status' in value);
}
// ---- Resolve base URL -----------------------------------------------------
function resolveBaseUrl(override) {
    if (override)
        return override.replace(/\/+$/, '');
    // Vite projects — access import.meta.env safely
    try {
        const meta = import.meta;
        const env = meta['env'];
        const viteUrl = env?.['VITE_EXAMPEN_API_URL'];
        if (viteUrl)
            return viteUrl.replace(/\/+$/, '');
    }
    catch {
        // import.meta.env unavailable outside Vite
    }
    // Node / generic — access process.env safely
    try {
        const g = globalThis;
        const proc = g['process'];
        const nodeUrl = proc?.env['EXAMPEN_API_URL'];
        if (nodeUrl)
            return nodeUrl.replace(/\/+$/, '');
    }
    catch {
        // process unavailable in browser
    }
    return '';
}
// ---- Retry helper ---------------------------------------------------------
const DEFAULT_MAX_RETRIES = 2;
const RETRY_DELAY_MS = 500;
async function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
// ---- API request ----------------------------------------------------------
export async function apiRequest(method, path, options, config) {
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
    const headers = {
        'Content-Type': 'application/json',
        ...(config?.defaultHeaders ?? {}),
        ...(token ? getAuthHeaders(token) : {}),
        ...(options?.headers ?? {}),
    };
    let init = {
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
    let lastError;
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
                code: errorBody.code ?? 'UNKNOWN',
                message: errorBody.message ?? response.statusText,
                status: response.status,
            };
            throw lastError;
        }
        const data = (await response.json());
        return { data, status: response.status };
    }
    // Should not reach here, but satisfies the type checker
    throw lastError ?? { code: 'RETRY_EXHAUSTED', message: '503 retries exhausted', status: 503 };
}
// ---- Convenience methods --------------------------------------------------
export function apiGet(path, options, config) {
    return apiRequest('GET', path, options, config);
}
export function apiPost(path, body, options, config) {
    return apiRequest('POST', path, { ...options, body }, config);
}
export function apiPatch(path, body, options, config) {
    return apiRequest('PATCH', path, { ...options, body }, config);
}
/**
 * Open a WebSocket connection for live invigilator feeds.
 * Returns the WebSocket instance so the caller can close it.
 */
export function connectWs(opts) {
    const baseUrl = resolveBaseUrl(opts.config?.baseUrl);
    const wsBase = baseUrl.replace(/^http/, 'ws');
    const wsUrl = opts.url ?? `${wsBase}${opts.path ?? '/api/v1/invigilator/ws'}`;
    const tokenVal = opts.token ?? getToken();
    const fullUrl = tokenVal ? `${wsUrl}?token=${encodeURIComponent(tokenVal)}` : wsUrl;
    const ws = new WebSocket(fullUrl);
    ws.onmessage = (event) => {
        const envelope = JSON.parse(String(event.data));
        opts.onMessage(envelope);
    };
    if (opts.onError)
        ws.onerror = opts.onError;
    if (opts.onClose)
        ws.onclose = opts.onClose;
    return ws;
}
//# sourceMappingURL=api-client.js.map