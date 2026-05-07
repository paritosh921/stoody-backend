# Chapter 10: Student BFF and Portal

## Status
- **Build status:** ACTIVE — substantially implemented in `frontend/src/components/exam-pen/ExamPenStudent.tsx`
- **Authority source:** `integration/STOODY_INTEGRATION_SPEC.md`

## Overview

Student-facing ExamPen behavior is read-oriented. Students view published outcomes, objection status, and supporting feedback through Stoody-facing surfaces after DCR or PCR results are finalized.

```text
╔══════════════════════════════════════════════════════════════╗
║                    EXAMPEN STUDENT WEB UI                   ║
╠══════════════════════════════════════════════════════════════╣
║ Published Exam List                                         ║
║   ├─ Score Card / Breakdown                                 ║
║   ├─ Per-Question Score + Reference Answer                  ║
║   ├─ Recheck Request Dialog                                 ║
║   ├─ Teacher Feedback                                       ║
║   └─ Student-Teacher Conversation                           ║
╚══════════════════════════════════════════════════════════════╝
```

## Implementation Status

The following student surfaces are implemented in `frontend/src/components/exam-pen/`:

| Section | Component | Status | Notes |
|---|---|---|---|
| Published exam list | `ExamPenStudent.tsx` | **Built** | Fetches via `exampenAPI.getStudentExamsV2()` |
| Score card with total + percentage | `EnhancedScoreCard` (inline) | **Built** | Expands to per-question breakdown |
| Per-question breakdown | Inline in `EnhancedScoreCard` | **Built** | Shows score, max score, reference answer, confidence, recheck status badge |
| Recheck request | `RecheckRequestDialog.tsx` | **Built** | Per-question dialog for requesting re-evaluation |
| Recheck status badges | Shared `RECHECK_STATUS` from `examPenStatus.ts` | **Built** | open, under_review, resolved_* |
| Teacher recheck response | Inline in question breakdown | **Built** | Shows teacher response text when recheck is resolved |
| Conversation threads | `StudentConversationList.tsx` | **Built** | Thread list + detail with message bubbles, send, resolve |
| Status module | `examPenStatus.ts` | **Built** | `RECHECK_STATUS`, `CONVERSATION_STATUS` shared maps |

**Backend gaps (UI ahead of mounted routes):**
- `exampenAPI.getStudentExamsV2()` — endpoint must be confirmed mounted
- `exampenAPI.getExamScoresV2()` — endpoint must be confirmed mounted
- `exampenAPI.createRecheckRequest()` — backend recheck router not yet mounted
- `exampenAPI.getConversationThreads()` — backend conversation router not yet mounted

## Student Implementation Sections

| Section | Purpose | Primary UI responsibility | Key boundary |
|---|---|---|---|
| Published exam list | Show only visible student outcomes | Exam cards or list of published ExamPen results | Must not expose unpublished engine drafts |
| Score detail | Show total score and per-question breakdown | Student-readable score card | Must derive from finalized/published outputs |
| Recheck visibility | Show whether a response/result is under objection or recheck | Badges, status panels, resolution summary | Student sees policy-approved visibility only |
| Feedback view | Show supporting explanation or teacher-published commentary | Feedback text and score explanation | No raw evaluator internals unless policy allows |
| Student-teacher conversation | Let student follow discussion about a recheck or objection | Linked thread per exam/result item | Depends on chat + objection workflow |

## Suggested Student Information Architecture

```text
ExamPen Student
├── Published Exams
│   ├── Score Summary
│   ├── Per-question Breakdown
│   ├── Feedback
│   └── Recheck Status
└── Conversations
    ├── Open Recheck Threads
    └── Resolved Threads
```

## Section Breakdown

### 1. Published Exam List

| Sub-component | What the frontend should provide | Notes |
|---|---|---|
| Published exam cards | Show visible exams only after publication | No draft leakage |
| Summary metadata | Title, exam type, total score, publish date | Quick scan surface |
| Entry into score detail | Click one exam to open detailed breakdown | Student drilldown |

### 2. Score Detail

| Sub-component | What the frontend should provide | Notes |
|---|---|---|
| Total score summary | Overall score and max score | Student-first presentation |
| Per-question breakdown | Score by question with feedback where allowed | Needs published detail support |
| Feedback panel | Show released feedback or explanation | Must respect review policy |

### 3. Recheck / Flag Visibility

| Sub-component | What the frontend should provide | Notes |
|---|---|---|
| Recheck status badge | Open, under review, resolved, changed, unchanged | Student-facing lifecycle |
| Objection summary | Why the student raised concern and what happened | Policy-filtered view |
| Resolution outcome | Score changed vs no change | Final communication surface |

### 4. Student-Teacher Conversation

| Sub-component | What the frontend should provide | Notes |
|---|---|---|
| Conversation list | Show all recheck-linked discussion threads | Student-centric entry point |
| Thread view | Show messages tied to one exam or result item | Must preserve context |
| Context binding | Link thread to exam, question, or score item | Avoid detached chat |

## Explicit Student Boundary

| Concern | Student portal owns | Student portal does not own |
|---|---|---|
| Published results view | Yes | — |
| Draft engine output | No | teacher/review side only |
| Manual score override | No | teacher/admin review side |
| Canonical artifact visibility | No direct raw-artifact authority | teacher/review side only unless policy allows |
| Recheck conversation follow-up | Yes, if enabled by policy | — |

## Alignment Rules

1. Students do not read draft evaluator internals unless the review policy allows it.
2. Published results come from finalized engine outputs, not directly from raw artifacts.
3. Practice persistence stays in the existing Stoody backend path.
4. Student-facing recheck/conversation UI must remain subordinate to teacher review and publication policy.

## Related Docs

- `api/student-bff.openapi.yaml`
- `api/review.openapi.yaml`
- `integration/STOODY_INTEGRATION_SPEC.md`
- `chapters/14_OBJECTION_REVIEW.md`
- `chapters/17_CHAT_SYSTEM.md`
