import { describe, it, expect } from 'vitest';
import type {
  Exam,
  ExamStatus,
  Score,
  ScoreStatus,
  Objection,
  ObjectionStatus,
  User,
  StoodyRole,
  ExamPenRole,
  StudentBinding,
  QuestionScore,
} from '../src/types';
import type {
  HubStatus,
  PenSyncStatus,
  DongleStatus,
  UploadStatus,
  WsEventType,
  ChatMessage,
  MissIndicatorState,
  StudentExamViewStatus,
} from '../src/types-hub';

// ---------------------------------------------------------------------------
// Type assertion tests: objects conforming to interfaces compile correctly.
// These verify that the type definitions are structurally sound. If the
// interfaces change in a breaking way, these assignments fail at compile time.
// ---------------------------------------------------------------------------

describe('type assertions — core domain', () => {
  it('Exam interface accepts valid object', () => {
    const exam: Exam = {
      exam_id: 'e_001',
      title: 'Math Exam 1',
      subject_id: 's_math',
      class_id: 'c_10A',
      section_id: 'sec_A',
      scheduled_at: '2026-04-01T09:00:00Z',
      duration_min: 90,
      total_marks: 100,
      question_count: 10,
      state: 'created',
      created_by: 'u_tutor_001',
    };
    expect(exam.exam_id).toBe('e_001');
    expect(exam.state).toBe('created');
  });

  it('Score interface accepts valid object', () => {
    const qs: QuestionScore = {
      question_id: 'q_001',
      ai_score: 8,
      current_score: 9,
      max_score: 10,
      confidence: 0.92,
    };
    const score: Score = {
      exam_id: 'e_001',
      student_id: 'u_stu_001',
      total_score: 85,
      max_score: 100,
      lifecycle_state: 'ai_draft',
      questions: [qs],
    };
    expect(score.lifecycle_state).toBe('ai_draft');
  });

  it('Objection interface accepts valid object', () => {
    const obj: Objection = {
      objection_id: 'obj_001',
      exam_id: 'e_001',
      student_id: 'u_stu_001',
      question_id: 'q_003',
      status: 'filed',
      objection_text: 'My diagram was not recognized',
      filed_at: '2026-04-02T10:00:00Z',
    };
    expect(obj.status).toBe('filed');
  });

  it('User interface accepts valid object', () => {
    const user: User = {
      user_id: 'u_001',
      tenant_id: 't_001',
      stoody_role: 'tutor',
      exampen_roles: ['tutor', 'evaluator'],
      display_name: 'Dr. Anita Sharma',
    };
    expect(user.exampen_roles).toContain('evaluator');
  });

  it('StudentBinding interface accepts valid object', () => {
    const binding: StudentBinding = {
      exam_id: 'e_001',
      pen_mac: 'AA:BB:CC:DD:EE:FF',
      student_id: 'u_stu_001',
      status: 'confirmed',
      source: 'registration_scan',
      bound_at: '2026-04-01T08:30:00Z',
    };
    expect(binding.status).toBe('confirmed');
  });
});

describe('type assertions — hub and view models', () => {
  it('HubStatus interface accepts valid object', () => {
    const hub: HubStatus = {
      exam_id: 'e_001',
      state: 'timer_running',
      timer_remaining_sec: 2400,
      upload_status: 'pending',
    };
    expect(hub.state).toBe('timer_running');
  });

  it('ChatMessage interface accepts valid object', () => {
    const msg: ChatMessage = {
      message_id: 'm_001',
      sender_id: 'u_stu_001',
      recipient_id: 'u_tutor_001',
      exam_id: 'e_001',
      content: 'Question about Q3 scoring',
      sent_at: '2026-04-02T11:00:00Z',
    };
    expect(msg.content).toContain('Q3');
  });
});

// ---------------------------------------------------------------------------
// FSM state literal unions: reject invalid values at compile time.
// Each @ts-expect-error proves the union rejects the bad literal.
// ---------------------------------------------------------------------------

describe('FSM state unions reject invalid values', () => {
  it('ExamStatus rejects invalid state', () => {
    // @ts-expect-error 'exploded' is not assignable to ExamStatus
    const _bad: ExamStatus = 'exploded';
    expect(_bad).toBeDefined(); // runtime no-op; compile check above
  });

  it('ScoreStatus rejects invalid state', () => {
    // @ts-expect-error 'deleted' is not assignable to ScoreStatus
    const _bad: ScoreStatus = 'deleted';
    expect(_bad).toBeDefined();
  });

  it('ObjectionStatus rejects invalid state', () => {
    // @ts-expect-error 'ignored' is not assignable to ObjectionStatus
    const _bad: ObjectionStatus = 'ignored';
    expect(_bad).toBeDefined();
  });

  it('StoodyRole rejects invalid role', () => {
    // @ts-expect-error 'wizard' is not assignable to StoodyRole
    const _bad: StoodyRole = 'wizard';
    expect(_bad).toBeDefined();
  });

  it('ExamPenRole rejects invalid role', () => {
    // @ts-expect-error 'hacker' is not assignable to ExamPenRole
    const _bad: ExamPenRole = 'hacker';
    expect(_bad).toBeDefined();
  });

  it('PenSyncStatus rejects invalid state', () => {
    // @ts-expect-error 'exploded' is not assignable to PenSyncStatus
    const _bad: PenSyncStatus = 'exploded';
    expect(_bad).toBeDefined();
  });

  it('DongleStatus rejects invalid state', () => {
    // @ts-expect-error 'missing' is not assignable to DongleStatus
    const _bad: DongleStatus = 'missing';
    expect(_bad).toBeDefined();
  });

  it('UploadStatus rejects invalid state', () => {
    // @ts-expect-error 'corrupted' is not assignable to UploadStatus
    const _bad: UploadStatus = 'corrupted';
    expect(_bad).toBeDefined();
  });

  it('WsEventType rejects invalid event', () => {
    // @ts-expect-error 'hack.attempt' is not assignable to WsEventType
    const _bad: WsEventType = 'hack.attempt';
    expect(_bad).toBeDefined();
  });

  it('MissIndicatorState rejects invalid state', () => {
    // @ts-expect-error 'maybe' is not assignable to MissIndicatorState
    const _bad: MissIndicatorState = 'maybe';
    expect(_bad).toBeDefined();
  });

  it('StudentExamViewStatus rejects invalid state', () => {
    // @ts-expect-error 'deleted' is not assignable to StudentExamViewStatus
    const _bad: StudentExamViewStatus = 'deleted';
    expect(_bad).toBeDefined();
  });
});
