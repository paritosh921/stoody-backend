import type { ExamPenRole, StoodyRole } from './types';
/** Decoded JWT claims from a Stoody-issued token, enriched by svc-auth. */
export interface ExamPenClaims {
    user_id: string;
    tenant_id: string;
    stoody_role: StoodyRole;
    exampen_roles: ExamPenRole[];
    name: string;
    email: string;
    exp: number;
    iat: number;
}
/**
 * Build auth headers for an ExamPen API request.
 * Includes the Bearer token and a static CSRF header.
 */
export declare function getAuthHeaders(token: string): Record<string, string>;
/**
 * Decode the payload section of a JWT without verifying the signature.
 * Verification is the responsibility of svc-auth on the server side.
 *
 * Throws if the token is malformed or the payload is not valid JSON.
 */
export declare function parseJwtClaims(token: string): ExamPenClaims;
/**
 * Returns true when the token's `exp` claim is in the past (or missing).
 * Uses a 30-second grace buffer to account for clock skew.
 */
export declare function isTokenExpired(token: string): boolean;
/** Persist the bearer token to localStorage. */
export declare function storeToken(token: string): void;
/** Retrieve the stored bearer token, or null if absent. */
export declare function getToken(): string | null;
/** Remove the stored bearer token. */
export declare function clearToken(): void;
//# sourceMappingURL=auth.d.ts.map