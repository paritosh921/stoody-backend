# Chapter 09: Teacher BFF and Dashboard

## Status
- **Build status:** DRAFT
- **Authority source:** `integration/STOODY_INTEGRATION_SPEC.md`

## Overview

Teacher-facing surfaces read across the shared ingest substrate, DCR outputs, PCR outputs, review state, and analytics. They aggregate; they do not become a second evaluator.

```text
Stoody teacher session
        │
        ▼
   teacher BFF / dashboard
        │
        ├-> exam orchestration
        ├-> ingest status
        ├-> DCR results
        ├-> PCR results / flags
        └-> review + analytics
```

## Web Frontend Implementation Map

This chapter describes the **web frontend implementation components** that belong in `frontend/` for teacher/admin ExamPen work.

It is an **implementation breakdown**, not a contract source:

- route shape still belongs to `api/*.openapi.yaml`
- storage and lifecycle still belong to root architecture and governance docs
- mobile-only live collection flows remain outside this chapter's ownership

```text
╔══════════════════════════════════════════════════════════════╗
║                  EXAMPEN TEACHER / ADMIN WEB UI             ║
╠══════════════════════════════════════════════════════════════╣
║ Exam List                                                   ║
║   ├─ Paper Setup                                            ║
║   ├─ Invigilator Setup                                      ║
║   ├─ Collection Monitor                                     ║
║   ├─ Response Review + AI Analysis                          ║
║   ├─ Score Override / Publish                               ║
║   └─ Student Follow-up                                      ║
╚══════════════════════════════════════════════════════════════╝
```

## Actionable Implementation Sections

| Section | Purpose | Primary UI responsibility | Key dependency boundary |
|---|---|---|---|
| Paper Setup | Prepare a conducted exam before collection starts | PDF preview, question inspection, setup checklist, region mapping handoff | Reads prepared exam metadata; does not write canonical artifacts |
| Invigilator Setup | Hand off an exam session to invigilator-side flows | Generate invigilator code, show expiry/instructions/session readiness | Web generates and explains; mobile/hub consumes |
| Collection Monitor | Show what has been collected and what is still missing | Submission progress, student submission status, upload/segmentation state | Reads ingest and orchestration state only |
| Response Viewer | Let teacher inspect collected pages or submission artifacts | Student list, submission drilldown, page/image thumbnails, source markers | Must remain read-only over canonical artifacts |
| AI Analysis + Question View | Show question context beside engine output | Question text, student response, extracted text, confidence, feedback, flags | Displays engine results; does not re-evaluate independently |
| Recheck Flags | Triage blocked or suspicious responses | Pending/blocked/ready queues, teacher decisions, resolution notes | Review actions must follow review policy |
| Score Editing | Allow controlled human correction | Override dialog, publish controls, audit visibility | Manual change must remain auditable |
| Student Clickable List | Make one-student drilldown easy | Student table/list with deep navigation into one exam attempt | Pure reader over scoped teacher-visible students |
| Student-Teacher Conversation | Resolve score/recheck discussions | Linked discussion thread per exam / question / response | Depends on chat and objection/review policy |
| Analytics | Operational and result summaries for one exam | Submission counts, blocked counts, published counts, throughput summaries | Aggregation only; not a second scoring engine |

## Suggested Teacher/Admin Information Architecture

```text
ExamPen Teacher/Admin
├── Exam List
├── Paper Setup
│   ├── PDF Preview
│   ├── Question List
│   ├── Setup Checklist
│   └── Question Region Mapping
├── Invigilator Setup
│   ├── Invigilator Code
│   ├── Expiry / Handoff
│   └── Session Ready State
├── Collection Monitor
│   ├── Student Submission List
│   ├── Upload / Segmentation Status
│   └── Submission Viewer
├── Review Queue
│   ├── Pending
│   ├── Blocked
│   └── Ready to Publish
├── Results
│   ├── Student Table
│   ├── Response Detail
│   ├── Score Override
│   └── Publish
└── Conversations
    ├── Open Rechecks
    └── Resolved Threads
```

## Section Breakdown

### 1. Paper Setup

| Sub-component | What the frontend should provide | Notes |
|---|---|---|
| Prepared exam list entry | Select a prepared exam before any student submission exists | Entry point into setup |
| Paper preview | Render the prepared PDF or equivalent paper view | Read-only preview |
| Question list | Show question order, type, and setup completeness | Needed before evaluation trust |
| Setup readiness checklist | Highlight missing answer-key, question-type, or mapping prerequisites | Prevent hidden partial setup |
| Question region mapping | Support region/bbox definition where structured DCR depends on it | This is the main known setup gap |

```text
Paper Setup
    │
    ├── select prepared exam
    ├── preview paper
    ├── verify questions
    ├── check setup completeness
    └── map question regions where required
```

### 2. Invigilator Setup

| Sub-component | What the frontend should provide | Notes |
|---|---|---|
| Invigilator code generation | Generate handoff code from the web teacher/admin surface | Web-owned |
| Expiry and handoff panel | Show code, expiry, and teacher-facing instructions | Avoid burying it inside a generic exam card |
| Session readiness state | Show whether the exam is ready to be picked up by invigilator/mobile flows | Lifecycle visibility only |

### 3. Collection Monitor

| Sub-component | What the frontend should provide | Notes |
|---|---|---|
| Student submission list | Show uploaded / missing / incomplete collection status per student | Must be easy to scan |
| Upload progress | Show ingestion / upload acknowledgement progress | Operational, not evaluative |
| Segmentation/evaluation state hints | Show pending / blocked / evaluated progression | Helps triage next action |

### 4. Response Viewer

| Sub-component | What the frontend should provide | Notes |
|---|---|---|
| Submission drilldown | Open one student's submission from the exam list or result table | Teacher-centric workflow |
| Page/image panel | Show page thumbnails and full selected artifact | Supports pen and camera sources |
| Source marker | Indicate whether the artifact came from pen or camera | Important for trust and fallback analysis |
| Page metadata | Show page number, upload time, segmentation state, and related question range | Supports auditability |

```text
Student List
    │
    └── one student
          ├── raw pages / images
          ├── source + page metadata
          ├── AI analysis
          ├── flags
          └── final score actions
```

### 5. AI Analysis + Question View

| Sub-component | What the frontend should provide | Notes |
|---|---|---|
| Question context panel | Show question text, type, expected answer/rubric context | Prevent detached scoring review |
| Student response evidence | Show extracted text, response image, or engine-visible evidence | Must be tied to the exact response |
| Engine summary | Distinguish DCR vs PCR results clearly | No hidden mixed interpretation |
| Confidence and flags | Show OCR confidence, segmentation confidence, blocking flags | Teacher needs actionable reasons |
| Feedback panel | Show overall feedback and score explanation | Read engine output directly |

### 6. Recheck Flags

| Sub-component | What the frontend should provide | Notes |
|---|---|---|
| Pending queue | Responses awaiting downstream evaluation or review | Work queue |
| Blocked queue | Responses blocked by flags or confidence issues | Main teacher review surface |
| Ready-to-publish queue | Responses that have cleared review | Publication staging |
| Resolution actions | Accept, reject, manual score, add rationale | Must remain policy-driven |
| Resolution visibility | Show why something was blocked and how it was resolved | Needed for later audit |

### 7. Score Editing

| Sub-component | What the frontend should provide | Notes |
|---|---|---|
| Student result table | Show per-student DCR/PCR/combined outcome | Primary result surface |
| Response-level override | Override one response/evaluation cleanly | Human correction path |
| Publish control | Publish reviewed output once ready | Final teacher action |
| Audit trail visibility | Show before/after score and reason | Strongly recommended even if not fully built yet |

### 8. Student Clickable List

| Sub-component | What the frontend should provide | Notes |
|---|---|---|
| Student table/list | Show clickable roster within one exam | Fast drilldown surface |
| Per-student state | Missing, uploaded, blocked, reviewed, published | Lifecycle summary |
| One-student exam view | Consolidate pages, analysis, flags, score, and conversation for one student | Best deep-review unit |

### 9. Student-Teacher Conversation

| Sub-component | What the frontend should provide | Notes |
|---|---|---|
| Recheck thread | Message thread tied to exam/result/recheck | Depends on chat system support |
| Response-linked discussion | Link a message to a flagged response or score item | Avoid context loss |
| Resolution state | Open, teacher replied, resolved, score changed, no change | Should align with objection workflow |

## Explicit Web Boundary

| Concern | Web frontend owns | Web frontend does not own |
|---|---|---|
| Paper preparation | Yes | — |
| Invigilator code generation | Yes | — |
| Live multi-hub pairing | No | mobile / hub side |
| On-device BLE control | No | mobile / hub side |
| Review queue and score override | Yes | — |
| Canonical artifact writes | No | ingest substrate |
| Platform-wide admin fleet controls | No | super-admin |

## Alignment Rules

1. Teacher surfaces do not write canonical raw artifacts.
2. Teacher surfaces do not bypass the review or gate rules.
3. Tutor visibility follows the existing admin-owned student visibility model.
4. Web-only flows must not absorb mobile-only live invigilation responsibilities.
5. Response review UI must stay attached to canonical ingest outputs and engine outputs, not teacher-entered stand-ins.

## Teacher Workspace Implementation Brief

This section is the **frontend implementation brief** for the AI agent that will build the teacher exam UI inside the existing ExamPen exam tab.

### Product Direction

Do **not** move raw paper upload, OCR, question editing, DCR/PCR selection, DCR template upload, or paper finalization into the teacher exam tab.

Keep this split:

| Area | Owns |
|---|---|
| Document Manager | Upload PDF, OCR, question editing, DCR/PCR mode, DCR template, finalize paper |
| Teacher Exam Tab | Select finalized exam, invigilator setup, collection monitoring, student review workspace, scoring, publish, recheck |

### Replace the Current Top-Level Teacher Tabs

The current top-level tabs (`Exam List`, `Exam Setup`, `Review Queue`, `Results`) are not clear enough for teacher workflow.

Recommended top-level tab model:

| Tab | Purpose |
|---|---|
| `Exams` | Choose exam and see operational status |
| `Workspace` | Main teacher review workspace for one selected exam |
| `Results` | Exam-wide final scores, overrides, publish state |
| `Recheck` | Flagged follow-up, objections, and linked discussion |

`Workspace` replaces the need for separate top-level `Exam Setup` and `Review Queue` views.

### Core Workspace Layout

The main teacher experience should be a **single IDE-style workspace**, not scattered cards.

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│ Header: exam title | type | lifecycle | invigilator | counts | actions      │
├───────────────┬───────────────────────────────┬───────────────────────────────┤
│ Student List  │ Question Paper + Student Copy │ Question Score / AI Inspector │
│               │                               │                               │
│ search        │ left: paper PDF               │ per-question score cards      │
│ filters       │ right: scans/strokes pages    │ score / answer / confidence   │
│ student rows  │                               │ override / review actions     │
└───────────────┴───────────────────────────────┴───────────────────────────────┘
```

Interaction model:

- left pane behaves like an explorer
- center behaves like the working editor surface
- right pane behaves like an inspector / problem panel

### Required Workspace Behavior

| Interaction | Expected result |
|---|---|
| Select exam | Open the workspace for that exam |
| Select student | Load that student's pages/strokes and score context |
| Select question | Focus question paper + answer evidence + question score card |
| Scroll copy pages | Inspect full response context without leaving the page |
| Open blocked item | Jump directly into the same workspace context |
| Override score | Act from the inspector without leaving the workspace |

### Workspace Sections

#### 1. Header Strip

Show:

- exam title
- DCR/PCR type
- lifecycle state
- question count
- student count
- blocked count
- invigilator setup action
- refresh action
- jump-to-results action

#### 2. Left Pane: Student Explorer

Show one searchable student list for the selected exam.

Each student row should show:

- student name
- student identifier
- submission status
- badges for `missing`, `uploaded`, `evaluating`, `blocked`, `ready`, `published`
- source tag such as `pen`, `camera`, or `mixed`
- count of submitted pages where available

#### 3. Center Pane: Dual Viewer

The center area must show both the paper and the selected student's submission.

| Center sub-pane | Purpose |
|---|---|
| Question Paper Viewer | Prepared paper PDF with question navigation |
| Student Copy Viewer | Scrollable scans/strokes pages for the selected student |

Rules:

- question paper remains visible while reviewing a student
- student pages must be scrollable
- page thumbnails or page index must exist
- if question-region mapping exists, clicking a question should focus the relevant answer area
- if mapping is unavailable, the workspace must still function at full-page level

#### 4. Right Pane: Question Score / AI Inspector

This is the primary review surface and must be first-class.

Every question should have a score card that shows:

- question number
- short question context
- expected answer or rubric context where available
- student detected answer / extracted text
- AI score
- max score
- confidence / engine info
- flags / reasons
- override action
- accept / recheck / review action as allowed

This right-side inspector should feel like a code editor inspector panel rather than a dashboard tile list.

### Screen States

| State | Expected UI |
|---|---|
| No exam selected | Show exam chooser / empty workspace |
| Exam selected, no student | Show paper + exam summary + student list; inspector prompts for student selection |
| Student selected, no question | Show student pages + all question score cards |
| Student + question selected | Highlight question, focus relevant score card, and sync the relevant answer region/page if available |

### Build Phases For AI Agent

| Phase | Scope |
|---|---|
| 1 | Replace tabs with `Exams`, `Workspace`, `Results`, `Recheck` |
| 2 | Build the empty `Workspace` shell and header |
| 3 | Add student explorer pane |
| 4 | Add question paper viewer + student copy viewer |
| 5 | Add question score / AI inspector cards |
| 6 | Surface invigilator setup inside workspace header or panel |
| 7 | Connect blocked/recheck flows back into workspace context |

### File-Level Handoff

| File | Responsibility |
|---|---|
| `frontend/src/components/exam-pen/ExamPenTeacher.tsx` | Replace current tabs and host the workspace shell |
| `frontend/src/components/exam-pen/ExamList.tsx` | Continue as `Exams` tab entry list |
| `frontend/src/components/exam-pen/PreparedExamView.tsx` | Reuse paper preview portions only; do not keep as the main teacher workflow |
| `frontend/src/components/exam-pen/ReviewQueue.tsx` | Reposition toward `Recheck` or workspace jump-in flows |
| `frontend/src/components/exam-pen/ExamResults.tsx` | Continue as `Results` tab |
| `frontend/src/components/exam-pen/ScoreOverride.tsx` | Reuse from workspace and results |
| `frontend/src/stores/examPenStore.ts` | Extend for selected student, question, page, and submission context |

New components expected:

- `TeacherWorkspace.tsx`
- `WorkspaceHeader.tsx`
- `StudentExplorerPane.tsx`
- `QuestionPaperPane.tsx`
- `StudentCopyPane.tsx`
- `QuestionInspectorPane.tsx`
- `QuestionScoreCard.tsx`
- `InvigilatorSetupPanel.tsx`
- `EmptyWorkspaceState.tsx`

### Acceptance Criteria

The implementation is acceptable when:

1. A teacher can select an exam and land in a single coherent workspace.
2. The workspace shows:
   - student list on one side
   - question paper in view
   - selected student's scrollable copy scans/strokes pages
   - per-question score / answer / AI cards in an inspector
3. Top-level teacher tab meanings are clearer than before.
4. The teacher can review one student without jumping across disconnected pages.
5. The exam tab remains a consumer of finalized papers rather than a second document manager.

## Related Docs

- `integration/STOODY_INTEGRATION_SPEC.md`
- `api/teacher-bff.openapi.yaml`
- `api/review.openapi.yaml`
- `chapters/14_OBJECTION_REVIEW.md`
- `chapters/10_STUDENT_BFF_PORTAL.md`
- `chapters/11_INVIGILATOR_CONSOLE.md`
- `chapters/17_CHAT_SYSTEM.md`
