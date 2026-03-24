import type {
  StudentExamCard,
  StudentScoreView,
  AnswerInsight,
  StudentObjection,
  CreateStudentObjectionRequest,
  Message,
  SendMessageRequest,
  PerformanceView,
  ListResponse,
} from "@/types/api";

// ─── HTTP helpers ───

const BASE = "/api/v1/student";

let _tokenAccessor: () => string = () => "";

/** Called once at app init so every request carries the JWT. */
export function setTokenAccessor(fn: () => string): void {
  _tokenAccessor = fn;
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${_tokenAccessor()}`,
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

// ─── Exams ───

export function fetchExams(): Promise<ListResponse<StudentExamCard>> {
  return request<ListResponse<StudentExamCard>>("/exams");
}

// ─── Scores ───

export function fetchScores(examId: string): Promise<StudentScoreView> {
  return request<StudentScoreView>(`/exams/${examId}/score`);
}

// ─── Answer insight ───

export function fetchAnswerInsight(
  examId: string,
  questionId: string,
): Promise<AnswerInsight> {
  return request<AnswerInsight>(
    `/exams/${examId}/questions/${questionId}/answer`,
  );
}

// ─── Objections ───

export function fetchObjections(): Promise<ListResponse<StudentObjection>> {
  return request<ListResponse<StudentObjection>>("/objections");
}

export function createObjection(
  body: CreateStudentObjectionRequest,
): Promise<StudentObjection> {
  return request<StudentObjection>(`/exams/${body.exam_id}/objections`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// ─── Chat ───

export function fetchChat(
  examId: string,
  teacherId: string,
): Promise<ListResponse<Message>> {
  return request<ListResponse<Message>>(
    `/exams/${examId}/chat/${teacherId}`,
  );
}

export function sendMessage(
  examId: string,
  teacherId: string,
  body: SendMessageRequest,
): Promise<Message> {
  return request<Message>(`/exams/${examId}/chat/${teacherId}`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// ─── Performance ───

/** Fetch combined history + strengths/weaknesses from /performance/history. */
export function fetchPerformanceHistory(): Promise<PerformanceView> {
  return request<PerformanceView>("/performance/history");
}

/** Fetch trend data for charts from /performance/trends. */
export function fetchPerformanceTrends(): Promise<{ history: Array<{ exam_id: string; score: number; percentile: number }> }> {
  return request<{ history: Array<{ exam_id: string; score: number; percentile: number }> }>("/performance/trends");
}

/** Fetch AI-generated strengths/weaknesses from /performance/strengths. */
export function fetchPerformanceStrengths(): Promise<{ strengths: string[]; weaknesses: string[] }> {
  return request<{ strengths: string[]; weaknesses: string[] }>("/performance/strengths");
}

/**
 * @deprecated Use fetchPerformanceHistory() instead.
 * Kept for backward compatibility with PerformancePage.
 */
export function fetchPerformance(): Promise<PerformanceView> {
  return fetchPerformanceHistory();
}
