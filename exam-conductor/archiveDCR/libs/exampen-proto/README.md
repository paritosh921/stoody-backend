# exampen-proto

Shared type definitions for the ExamPen system. This package provides both Python (Pydantic v2) and TypeScript interface definitions covering all domain models, NATS event envelopes, and BFF view types.

Every type is derived from the authoritative OpenAPI specs in `new-docs/api/` and NATS event schemas in `new-docs/contracts/events/`.

## Package Structure

```
exampen-proto/
├── python/
│   ├── pyproject.toml
│   └── exampen_proto/
│       ├── __init__.py       # Re-exports all models
│       ├── enums.py          # All FSM states and string enums
│       ├── exam.py           # Exam, lifecycle, bindings, roster
│       ├── stroke.py         # Raw chunks, processed strokes, upload status
│       ├── score.py          # Score projections, overrides, audit history
│       ├── user.py           # Auth claims, profiles, revocations
│       ├── page.py           # Page images, copy uploads, miss indicators
│       ├── ai.py             # AI results, confidence, answer insights
│       ├── plagiarism.py     # Flags, evidence, verdicts
│       ├── objection.py      # Filing, resolution, escalation
│       ├── analytics.py      # Leaderboard, class stats, performance
│       ├── chat.py           # Messages, threads, read receipts
│       ├── hub.py            # Dongle, pen sync, session, WebSocket
│       ├── bff.py            # Teacher and student BFF view models
│       └── events.py         # All NATS event envelope models
└── typescript/
    ├── package.json
    ├── tsconfig.json
    └── src/
        ├── index.ts          # Re-exports all types
        ├── enums.ts          # String literal union types
        ├── exam.ts
        ├── stroke.ts
        ├── score.ts
        ├── user.ts
        ├── page.ts
        ├── ai.ts
        ├── plagiarism.ts
        ├── objection.ts
        ├── analytics.ts
        ├── chat.ts
        ├── hub.ts
        ├── bff.ts
        └── events.ts
```

## Python Usage

Install as an editable local dependency:

```bash
pip install -e libs/exampen-proto/python
```

Import models directly from the top-level package:

```python
from exampen_proto import ExamDetail, ExamState, StrokeRawEvent

# Create an exam summary
exam = ExamSummary(
    exam_id="550e8400-e29b-41d4-a716-446655440000",
    subject_id="math-10",
    class_id="10-A",
    scheduled_at=datetime.now(),
    state=ExamState.CREATED,
)

# Validate a score event
event = ScoreUpdatedEvent(
    event_id="evt-1",
    occurred_at=datetime.now(),
    exam_id="550e8400-e29b-41d4-a716-446655440000",
    student_id="stu-42",
    lifecycle_state=ScoreLifecycleState.AI_DRAFT,
    total_score=78.5,
    reason=ScoreEventType.AI_DRAFT_CREATED,
)
```

Or import from specific modules:

```python
from exampen_proto.enums import ScoreLifecycleState
from exampen_proto.score import StudentScoreDetail
from exampen_proto.events import ExamLifecycleEvent
```

## TypeScript Usage

Reference via workspace path or install locally:

```json
{
  "dependencies": {
    "@exampen/proto": "file:../../libs/exampen-proto/typescript"
  }
}
```

Import types:

```typescript
import type {
  ExamDetail,
  ExamState,
  StrokeRawEvent,
  ScoreLifecycleState,
} from "@exampen/proto";

const exam: ExamDetail = {
  exam_id: "550e8400-e29b-41d4-a716-446655440000",
  subject_id: "math-10",
  class_id: "10-A",
  section_id: "A",
  title: "Mid-term Mathematics",
  scheduled_at: "2026-03-20T09:00:00Z",
  duration_min: 120,
  state: "created",
  total_marks: 100,
  question_count: 10,
  created_by: "tutor-1",
};
```

## Key FSM States

| Domain | States |
|--------|--------|
| Exam lifecycle | `created > armed > timer_running > sync_pending > scoring > finalized > published > locked` (+ `cancelled`) |
| Score lifecycle | `ai_draft > teacher_reviewed > finalized > published > objection_window > locked` |
| Objection lifecycle | `filed > assigned > reviewing > resolved` (+ `escalated`) |
| Pen binding | `provisional > confirmed` (+ `rejected`) |
| Plagiarism verdict | `pending > confirmed_plagiarism` / `dismissed` |

## Validation Level

This package targets **L2** (typecheck/lint verified). No runtime tests are included; correctness is enforced by Pydantic v2 strict mode (Python) and TypeScript strict compilation.
