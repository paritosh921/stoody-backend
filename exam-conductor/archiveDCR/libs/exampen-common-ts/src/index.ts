// ---------------------------------------------------------------------------
// Public barrel export for @exampen/common-ts
// ---------------------------------------------------------------------------

// ---- Core domain types ----------------------------------------------------
export type {
  ExamStatus,
  Exam,
  ExamVariant,
  Rubric,
  RubricStep,
  QuestionRegion,
  ScoreStatus,
  Score,
  QuestionScore,
  StepScore,
  ScoreOverride,
  ScoreEventType,
  ScoreHistoryItem,
  ObjectionStatus,
  ObjectionResolution,
  Objection,
  StoodyRole,
  ExamPenRole,
  User,
  StudentBinding,
} from './types';

// ---- Hub, analytics, chat, view types -------------------------------------
export type {
  LeaderboardEntry,
  PercentileData,
  ClassStats,
  QuestionDifficulty,
  HubSessionState,
  DongleStatus,
  PenSyncStatus,
  UploadStatus,
  HubStatus,
  DongleInfo,
  PenSyncInfo,
  MissIndicatorState,
  MissIndicatorCell,
  ChatMessage,
  ChatThread,
  WsEventType,
  WebSocketEnvelope,
  StudentExamViewStatus,
  StudentExamCard,
} from './types-hub';

// ---- Auth -----------------------------------------------------------------
export {
  getAuthHeaders,
  parseJwtClaims,
  isTokenExpired,
  storeToken,
  getToken,
  clearToken,
} from './auth';
export type { ExamPenClaims } from './auth';

// ---- API Client -----------------------------------------------------------
export {
  apiRequest,
  apiGet,
  apiPost,
  apiPatch,
  connectWs,
  isApiError,
} from './api-client';
export type {
  ApiResponse,
  ApiError,
  HttpMethod,
  RequestOptions,
  RequestInterceptor,
  ResponseInterceptor,
  ApiClientConfig,
  WsClientOptions,
} from './api-client';
