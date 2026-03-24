// ---------------------------------------------------------------------------
// Auth helpers: JWT decode (no verification), token storage, auth headers.
// ---------------------------------------------------------------------------
// ---- Header helpers -------------------------------------------------------
const CSRF_HEADER = 'X-Requested-With';
const CSRF_VALUE = 'ExamPen';
/**
 * Build auth headers for an ExamPen API request.
 * Includes the Bearer token and a static CSRF header.
 */
export function getAuthHeaders(token) {
    return {
        Authorization: `Bearer ${token}`,
        [CSRF_HEADER]: CSRF_VALUE,
    };
}
// ---- JWT helpers (client-side, no verification) ---------------------------
/**
 * Decode the payload section of a JWT without verifying the signature.
 * Verification is the responsibility of svc-auth on the server side.
 *
 * Throws if the token is malformed or the payload is not valid JSON.
 */
export function parseJwtClaims(token) {
    const parts = token.split('.');
    if (parts.length !== 3) {
        throw new Error('Malformed JWT: expected 3 dot-separated parts');
    }
    const payloadB64 = parts[1]; // length check above guarantees index 1 exists
    const json = decodeBase64Url(payloadB64);
    const raw = JSON.parse(json);
    if (typeof raw !== 'object' || raw === null) {
        throw new Error('JWT payload is not a JSON object');
    }
    return raw;
}
/**
 * Returns true when the token's `exp` claim is in the past (or missing).
 * Uses a 30-second grace buffer to account for clock skew.
 */
export function isTokenExpired(token) {
    const GRACE_SEC = 30;
    try {
        const claims = parseJwtClaims(token);
        if (typeof claims.exp !== 'number')
            return true;
        const nowSec = Math.floor(Date.now() / 1000);
        return nowSec >= claims.exp - GRACE_SEC;
    }
    catch {
        return true;
    }
}
// ---- Token storage (localStorage) -----------------------------------------
const STORAGE_KEY = 'exampen_token';
/** Persist the bearer token to localStorage. */
export function storeToken(token) {
    if (typeof localStorage === 'undefined')
        return;
    localStorage.setItem(STORAGE_KEY, token);
}
/** Retrieve the stored bearer token, or null if absent. */
export function getToken() {
    if (typeof localStorage === 'undefined')
        return null;
    return localStorage.getItem(STORAGE_KEY);
}
/** Remove the stored bearer token. */
export function clearToken() {
    if (typeof localStorage === 'undefined')
        return;
    localStorage.removeItem(STORAGE_KEY);
}
// ---- Internal utilities ---------------------------------------------------
/**
 * Decode a Base64-URL encoded string to a UTF-8 string.
 * Works in both browser (atob) and Node.js (Buffer) environments.
 */
function decodeBase64Url(input) {
    // Restore standard Base64 characters
    let base64 = input.replace(/-/g, '+').replace(/_/g, '/');
    // Pad to a multiple of 4
    const pad = base64.length % 4;
    if (pad === 2)
        base64 += '==';
    else if (pad === 3)
        base64 += '=';
    if (typeof atob === 'function') {
        return atob(base64);
    }
    // Node.js fallback — access Buffer via globalThis to avoid compile-time dep
    const g = globalThis;
    const BufferCtor = g['Buffer'];
    if (BufferCtor) {
        return BufferCtor.from(base64, 'base64').toString('utf-8');
    }
    throw new Error('No base64 decoder available (atob or Buffer)');
}
//# sourceMappingURL=auth.js.map