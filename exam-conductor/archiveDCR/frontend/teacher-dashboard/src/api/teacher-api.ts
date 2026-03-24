// Typed API functions for the Teacher BFF.
// Uses shared client from ./client.ts; types from ./teacher-types.ts.

import { get, post, patch, request } from './client';
import type {
  TeacherExamCard,
  ClassScoreRow,
  QuestionDetail,
  TeacherStudentDetail,
  ScoreOverridePayload,
  PlagiarismPreview,
  ObjectionInboxItem,
  ChatMessage,
  LeaderboardEntry,
  ClassStats,
  RubricQuestion,
  RubricPayload,
  QuestionRegion,
} from './teacher-types';

// Re-export all types so existing imports from '@/api/teacher-api' still work.
export type {
  TeacherExamCard,
  ClassScoreRow,
  QuestionDetail,
  TeacherStudentDetail,
  ScoreOverridePayload,
  MissState,
  MissIndicatorCell,
  MissIndicatorMatrix,
  PlagiarismPreview,
  ObjectionInboxItem,
  ChatMessage,
  LeaderboardEntry,
  ClassStats,
  RubricStep,
  RubricQuestion,
  RubricPayload,
  QuestionRegion,
} from './teacher-types';

export async function getExams(filters?: {
  subject_id?: string;
  class_id?: string;
}) {
  const params: Record<string, string> = {};
  if (filters?.subject_id) params.subject_id = filters.subject_id;
  if (filters?.class_id) params.class_id = filters.class_id;
  return get<{ items: TeacherExamCard[] }>(
    '/api/v1/teacher/exams',
    Object.keys(params).length ? params : undefined,
  );
}

export async function getExamDetail(examId: string) {
  return get<TeacherExamCard>(`/api/v1/teacher/exams/${examId}`);
}

export async function createExam(payload: Record<string, unknown>) {
  return post<TeacherExamCard>('/api/v1/teacher/exams', payload);
}

export async function getClassScores(examId: string) {
  return get<{ rows: ClassScoreRow[] }>(
    `/api/v1/teacher/exams/${examId}/scores`,
  );
}

export async function getStudentDetail(examId: string, studentId: string) {
  return get<TeacherStudentDetail>(
    `/api/v1/teacher/exams/${examId}/students/${studentId}`,
  );
}

export async function overrideScore(
  examId: string,
  studentId: string,
  payload: ScoreOverridePayload,
) {
  return patch<TeacherStudentDetail>(
    `/api/v1/teacher/exams/${examId}/students/${studentId}/questions/${payload.question_id}`,
    payload,
  );
}

export async function finalizeScores(examId: string) {
  return post<{ ok: boolean }>(
    `/api/v1/teacher/exams/${examId}/scores/finalize`,
    {},
  );
}

export async function publishScores(examId: string) {
  return post<{ ok: boolean }>(
    `/api/v1/teacher/exams/${examId}/scores/publish`,
    {},
  );
}

export async function getObjections(examId: string) {
  return get<{ items: ObjectionInboxItem[] }>(
    `/api/v1/teacher/exams/${examId}/objections`,
  );
}

export async function getObjectionDetail(objectionId: string) {
  return get<ObjectionInboxItem>(
    `/api/v1/teacher/objections/${objectionId}`,
  );
}

export async function resolveObjection(
  objectionId: string,
  payload: { verdict: string; new_score?: number; reason: string },
) {
  return post<ObjectionInboxItem>(
    `/api/v1/teacher/objections/${objectionId}/resolve`,
    payload,
  );
}

export async function escalateObjection(
  objectionId: string,
  payload: { target_role: string; reason: string },
) {
  return post<ObjectionInboxItem>(
    `/api/v1/teacher/objections/${objectionId}/escalate`,
    payload,
  );
}

export async function getLeaderboard(examId: string) {
  return get<{ items: LeaderboardEntry[] }>(
    `/api/v1/teacher/exams/${examId}/leaderboard`,
  );
}

export async function getClassStats(examId: string) {
  return get<ClassStats>(`/api/v1/teacher/exams/${examId}/class-stats`);
}

export async function getQuestionAnalysis(examId: string) {
  return get<{ items: QuestionDetail[] }>(
    `/api/v1/teacher/exams/${examId}/questions`,
  );
}

export async function getPlagiarismFlags(examId: string) {
  return get<{ items: PlagiarismPreview[] }>(
    `/api/v1/teacher/exams/${examId}/plagiarism`,
  );
}

export async function submitVerdict(
  flagId: string,
  payload: { verdict: string; reason: string },
) {
  return patch<PlagiarismPreview>(
    `/api/v1/teacher/plagiarism/${flagId}/verdict`,
    payload,
  );
}

// ---- Rubric / Regions / Upload / Assignment / Export / Chat ----

export async function getRubric(examId: string) {
  return get<{ questions: RubricQuestion[]; confidence_threshold: number }>(
    `/api/v1/teacher/exams/${examId}/rubric`,
  );
}

export async function saveRubric(examId: string, payload: RubricPayload) {
  return request<{ ok: boolean }>(
    'PUT',
    `/api/v1/teacher/exams/${examId}/rubric`,
    { body: JSON.stringify(payload), headers: { 'Content-Type': 'application/json' } },
  );
}

export async function getQuestionRegions(examId: string) {
  return get<{ regions: QuestionRegion[] }>(
    `/api/v1/teacher/exams/${examId}/regions`,
  );
}

export async function saveQuestionRegions(
  examId: string,
  regions: QuestionRegion[],
) {
  return request<{ ok: boolean }>(
    'PUT',
    `/api/v1/teacher/exams/${examId}/regions`,
    { body: JSON.stringify({ regions }), headers: { 'Content-Type': 'application/json' } },
  );
}

// Question paper upload requires S3/MinIO integration (not yet implemented
// in the teacher-BFF or svc-exam-orch). The upload flow will be:
// 1. POST to teacher-bff to get a presigned S3 upload URL
// 2. PUT directly to S3 from the browser
// 3. POST back to teacher-bff with the S3 key to link it to the exam
// For now, this is a no-op stub.
export async function uploadQuestionPaper(_examId: string, _file: File) {
  return { data: { upload_url: '' }, status: 501 };
}

export async function assignStaff(
  examId: string,
  invigilatorIds: string[],
  evaluatorIds: string[],
  doubleBlind = false,
) {
  // svc-exam-orch has ONE assignment endpoint that takes both roles
  return post<{ ok: boolean }>(
    `/api/v1/teacher/exams/${examId}/invigilators`,
    { invigilator_ids: invigilatorIds, evaluator_ids: evaluatorIds, double_blind: doubleBlind },
  );
}

export async function exportAnalyticsCsv(examId: string) {
  return post<{ export_url: string }>(
    `/api/v1/teacher/exams/${examId}/export`,
    { format: 'csv' },
  );
}

// Chat API calls svc-chat directly (not teacher-bff)
const CHAT_BASE = import.meta.env.VITE_CHAT_API_URL ?? '';

export async function getChatThread(examId: string, studentId: string) {
  return get<{ items: ChatMessage[] }>(
    `${CHAT_BASE}/api/v1/chat/threads/${examId}/${studentId}`,
  );
}

export async function sendChatMessage(
  examId: string,
  studentId: string,
  content: string,
) {
  return post<ChatMessage>(
    `${CHAT_BASE}/api/v1/chat/threads/${examId}/${studentId}`,
    { content },
  );
}
