import type { ExamStatus } from './types';
export interface LeaderboardEntry {
    rank: number;
    student_id: string;
    student_name?: string;
    score: number;
    percentile: number;
}
export interface PercentileData {
    exam_id: string;
    score: number;
    percentile: number;
}
export interface ClassStats {
    mean: number;
    median: number;
    std_dev: number;
    pass_rate: number;
    question_difficulty?: QuestionDifficulty[];
}
export interface QuestionDifficulty {
    question_id: string;
    avg_score: number;
}
export type HubSessionState = ExamStatus;
export type DongleStatus = 'healthy' | 'degraded' | 'failed';
export type PenSyncStatus = 'pending' | 'connecting' | 'syncing' | 'complete' | 'failed' | 'timeout';
export type UploadStatus = 'pending' | 'in_progress' | 'complete' | 'partial';
export interface HubStatus {
    exam_id: string;
    state: HubSessionState;
    timer_remaining_sec: number;
    upload_status: UploadStatus;
    backend_seen_at?: string;
}
export interface DongleInfo {
    dongle_mac: string;
    status: DongleStatus;
    connected_pens: number;
    capacity?: number;
}
export interface PenSyncInfo {
    pen_mac: string;
    student_id?: string;
    sync_status: PenSyncStatus;
    bytes_received?: number;
    total_chunks?: number;
}
export type MissIndicatorState = 'answered' | 'miss_no_strokes' | 'miss_sync_failure' | 'miss_pen_inactive' | 'not_attempted_confirmed';
export interface MissIndicatorCell {
    student_id: string;
    question_id: string;
    state: MissIndicatorState;
}
export interface ChatMessage {
    message_id: string;
    sender_id: string;
    recipient_id: string;
    exam_id: string;
    content: string;
    attachment_uri?: string;
    sent_at: string;
    read_at?: string;
}
export interface ChatThread {
    exam_id: string;
    other_user_id: string;
    messages: ChatMessage[];
}
export type WsEventType = 'session.snapshot' | 'sync.progress' | 'dongle.health' | 'upload.progress';
export interface WebSocketEnvelope {
    event_type: WsEventType;
    payload: Record<string, unknown>;
}
export type StudentExamViewStatus = 'upcoming' | 'scores_pending' | 'published' | 'objection_window_open' | 'locked';
export interface StudentExamCard {
    exam_id: string;
    title: string;
    subject_name?: string;
    scheduled_at: string;
    status: StudentExamViewStatus;
}
//# sourceMappingURL=types-hub.d.ts.map