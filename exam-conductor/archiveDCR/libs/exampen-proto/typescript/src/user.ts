/** User identity types: auth claims, profiles, role mappings, revocations. */

import type { ExamPenRole, StoodyRole, TokenStatus } from "./enums";

/** User profile information from Stoody. */
export interface Profile {
  display_name: string;
  email?: string;
  phone?: string;
  institute_name?: string;
}

/** Normalized ExamPen claims derived from a Stoody JWT. */
export interface NormalizedClaims {
  user_id: string;
  tenant_id: string;
  stoody_role: StoodyRole;
  exampen_roles: ExamPenRole[];
  token_source: "stoody_jwt";
  token_status: TokenStatus;
  subject_ids?: string[];
  class_ids?: string[];
  child_student_ids?: string[];
  profile: Profile;
}

/** Request to validate and normalize a Stoody JWT. */
export interface IntrospectRequest {
  token: string;
  expected_role?: ExamPenRole;
}

/** Request to revoke a token/session within ExamPen. */
export interface RevocationRequest {
  jti: string;
  subject_user_id?: string;
  reason: string;
  expires_at?: string;
}

/** Revocation state for a token JTI. */
export interface RevocationStatus {
  jti: string;
  revoked: boolean;
  revoked_at?: string;
  reason?: string;
}

/** Standard error response body. */
export interface ErrorResponse {
  code: string;
  message: string;
}
