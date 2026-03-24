# STOODY_INTEGRATION_SPEC.md
# ExamPen × Stoody — Platform Integration Specification

Reference: R4-EXAMPEN-DEVSTACK

---

## 1. Integration Model

ExamPen is a subsystem of the Stoody education platform. It does NOT replace Stoody. It plugs into Stoody's existing tutor and student experiences.

```
Stoody Platform
├── Tutor Login Portal (existing)
│   └── ExamPen Module (new tab/section)
│       ├── Exam management
│       ├── Score review
│       ├── AI analysis viewer
│       ├── Objection handling
│       └── Analytics
├── Student Login Portal (existing)
│   └── ExamPen Module (new tab/section)
│       ├── Score viewer
│       ├── Objection filing
│       ├── Chat with tutor
│       └── Historical performance
└── Mobile Apps
    ├── Tutor App (ExamPen: hub control + score management)
    └── Student App (ExamPen: score view + objection)
```

### 1.1 Authentication Integration

| Aspect | Approach |
|---|---|
| SSO | Stoody issues JWT. ExamPen `svc-auth` validates Stoody's JWT (shared signing key or JWKS endpoint). |
| User identity | Stoody's `user_id` is the primary key. ExamPen stores `stoody_user_id` in its user mapping table. |
| Roles | Stoody provides base role (tutor/student/parent). ExamPen adds exam-specific roles (invigilator, evaluator, reviewer) through `svc-auth` role mapping. |
| Session | Stoody login session extends to ExamPen. Stoody's bearer JWT is forwarded to ExamPen APIs; ExamPen does not issue a separate end-user session token. |

### 1.2 API Integration

| Direction | Method | Purpose |
|---|---|---|
| Stoody → ExamPen | REST API calls to ExamPen backend | Embed exam data in Stoody UI |
| ExamPen → Stoody | REST API calls to Stoody backend | Fetch student roster, class structure, subject mapping |
| Webhook | ExamPen → Stoody | Notify Stoody when scores are published (for Stoody's gradebook) |

### 1.3 Data Mapping

| Stoody Entity | ExamPen Entity | Sync Direction |
|---|---|---|
| Student profile | `student_id`, `name`, `class`, `section`, `roll` | Stoody → ExamPen (read-only) |
| Tutor profile | `teacher_id`, `name`, `subjects`, `classes` | Stoody → ExamPen (read-only) |
| Parent-child relation | `parent_user_id` → `student_id[]` | Stoody → ExamPen (read-only) |
| Class/Section | `class_id`, `section_id` | Stoody → ExamPen (read-only) |
| Subject | `subject_id`, `name` | Stoody → ExamPen (read-only) |
| Exam | Created in ExamPen, reference pushed to Stoody | ExamPen → Stoody (via webhook) |
| Score | Owned by ExamPen, summary pushed to Stoody gradebook | ExamPen → Stoody (via webhook) |

**Rule:** Stoody is the source of truth for student/tutor identity, class structure, and subject mapping. ExamPen NEVER creates students or classes — it only references Stoody's data.

---

## 2. Tutor Portal — Feature Matrix

### 2.1 Exam Lifecycle Features (Tutor Login → ExamPen Section)

| Feature | Description | Web Portal | Mobile App |
|---|---|---|---|
| **Create exam** | Define exam: subject, class/section, date, duration, question count, total marks, negative marking config, variants (A/B/C/D) | ✓ Full form | ✓ Simplified |
| **Define rubric** | Per-question: marks allocation, step breakdown, expected answer type (text/formula/diagram), auto-score confidence threshold | ✓ Rich editor | ✗ View only |
| **Define question regions** | Map question numbers to spatial regions on the answer sheet template. Upload answer sheet layout image, draw bounding boxes. | ✓ Visual editor | ✗ Not supported |
| **Assign invigilators** | Select tutors as invigilators for this exam. Generates rotating auth codes. | ✓ | ✓ |
| **Assign evaluators** | Select tutors as evaluators (may differ from invigilator). Double-blind assignment option. | ✓ | ✗ |
| **Set exam schedule** | Date, start time, duration, grace period, late-entry cutoff | ✓ | ✓ |
| **Manage question paper** | Upload question paper PDF per variant. Links to exam for student view post-exam. | ✓ | ✗ |
| **Register pens** | Trigger pen registration via hub (requires invigilator mode on mobile). View MAC→student mapping. Manual override for unregistered pens. | ✗ (Hub-dependent) | ✓ Invigilator mode |
| **Start/stop exam** | Arm timer, start countdown, emergency stop. Requires invigilator BLE connection to hub. | ✗ (Hub-dependent) | ✓ Invigilator mode |
| **Monitor sync** | Real-time: per-student sync progress, dongle health, hub status. | ✓ (via backend, delayed) | ✓ Real-time via BLE |
| **Trigger upload** | Send collected data from hub to backend. | ✗ | ✓ Invigilator mode |
| **Upload copy images** | Photograph physical answer copies as fallback. | ✗ | ✓ Camera capture |

### 2.2 Score Review Features

| Feature | Description | Web Portal | Mobile App |
|---|---|---|---|
| **Class score overview** | Table: all students, total score, AI confidence %, miss indicators, plagiarism flags. Sort/filter by any column. | ✓ Full table | ✓ Compact list |
| **Student drill-down** | Per-student: page-by-page answer images, per-question breakdown (AI score, manual score, step scores, confidence) | ✓ Side-by-side view | ✓ Scroll view |
| **AI analysis viewer** | For each question: recognized text, step detection output, confidence bars, original stroke rendering alongside AI interpretation | ✓ | ✓ Simplified |
| **Score edit** | Tap any question score → edit → mandatory reason → save. Shows old/new value. Audit-logged. | ✓ Inline edit | ✓ Inline edit |
| **Bulk approve** | Approve all AI-scored answers above confidence threshold in one action. Individually review below-threshold. | ✓ | ✗ |
| **Step-level marking** | View and edit marks at step level (formula: 2, substitution: 1, answer: 1). AI suggests, tutor confirms/overrides. | ✓ | ✓ |
| **Manual feedback entry** | Add text feedback per question per student. Visible to student in their portal. | ✓ Rich text | ✓ Text only |
| **Miss indicator review** | Grid view: all students × all questions. Color-coded: green/amber/red/gray. Click to drill down, view copy image, override status. | ✓ | ✓ |
| **Plagiarism review** | Flagged pairs list. Side-by-side answer comparison. Evidence: text similarity, temporal correlation, seating proximity. Confirm/dismiss with mandatory reason. | ✓ | ✓ Simplified |
| **Score finalization** | Lock scores for publication. Opens objection window. Sends notifications to students. | ✓ | ✓ |

### 2.3 Objection Handling Features

| Feature | Description | Web Portal | Mobile App |
|---|---|---|---|
| **Objection inbox** | List: student name, question #, objection text, status, date filed. Filter by status. | ✓ | ✓ |
| **Objection detail** | Side-by-side: student answer image + AI recognition + current score + rubric + student's objection text | ✓ | ✓ |
| **Approve objection** | Triggers re-score. Tutor enters new score + reason. | ✓ | ✓ |
| **Reject objection** | Mandatory reason. Student notified. | ✓ | ✓ |
| **Escalate** | Forward to HOD or senior evaluator. | ✓ | ✗ |
| **Chat with student** | Per-objection message thread. Tutor can send annotated answer images. | ✓ | ✓ |

### 2.4 Analytics Features

| Feature | Description | Web Portal | Mobile App |
|---|---|---|---|
| **Leaderboard** | Rank list: student name, score, percentile. Configurable scope (section/grade/institute). | ✓ | ✓ |
| **Historical performance** | Per-student trend: scores across exams over time. Subject-wise breakdown. | ✓ Charts | ✓ Simplified |
| **Class analytics** | Mean, median, std dev, pass %, question-wise difficulty analysis. | ✓ | ✓ Summary |
| **Export** | PDF report cards, CSV bulk export (scores, analytics), print-ready answer sheets. | ✓ | ✗ |
| **Question-wise analysis** | Per-question: avg score, % attempted, % correct, common errors flagged by AI. | ✓ | ✗ |

---

## 3. Student Portal — Feature Matrix

### 3.1 Exam View Features (Student Login → ExamPen Section)

| Feature | Description | Web Portal | Mobile App |
|---|---|---|---|
| **Upcoming exams** | List of scheduled exams with date, time, subject, duration. | ✓ | ✓ |
| **Exam instructions** | Pre-exam: pen instructions, dos/don'ts, answer sheet format. | ✓ | ✓ |
| **Past exams** | List of completed exams with status (scores pending / published / objection window open). | ✓ | ✓ |

### 3.2 Score View Features

| Feature | Description | Web Portal | Mobile App |
|---|---|---|---|
| **Score summary** | Total score, percentage, percentile (within section and grade). Pass/fail indicator. | ✓ | ✓ |
| **Question-wise breakdown** | Per-question: marks obtained, max marks, AI confidence, step breakdown (if step-marking enabled). | ✓ | ✓ |
| **Answer image viewer** | View rendered answer pages (from strokes or uploaded copy). See what the AI saw. | ✓ Full-size | ✓ Pinch-zoom |
| **AI analysis view** | Read-only: see AI's interpretation of handwriting, step detection, recognized text. Helps student understand scoring rationale. | ✓ | ✓ Simplified |
| **Feedback view** | Tutor's text feedback per question. | ✓ | ✓ |
| **Miss indicators** | Student sees: which questions were marked as attempted vs not-attempted. If "sync failure" or "possible miss", student understands why a question may show 0. | ✓ | ✓ |
| **Percentile chart** | Visual: where student stands relative to class. Historical trend across exams. | ✓ Charts | ✓ Simplified |

### 3.3 Objection Features

| Feature | Description | Web Portal | Mobile App |
|---|---|---|---|
| **File objection** | Select question, write objection text (why the score is wrong), submit. Only during objection window. | ✓ | ✓ |
| **Objection status** | Track: filed → under review → resolved (approved/rejected). Notification on status change. | ✓ | ✓ |
| **View resolution** | See tutor's response: approved (new score) or rejected (reason). | ✓ | ✓ |
| **Chat with tutor** | Per-objection message thread. Student can explain further, attach annotated screenshots. | ✓ | ✓ |

### 3.4 Historical Performance

| Feature | Description | Web Portal | Mobile App |
|---|---|---|---|
| **Score history** | All exams, all subjects. Table with sort/filter. | ✓ | ✓ |
| **Trend charts** | Score trend per subject over time. Percentile trend. | ✓ | ✓ |
| **Strength/weakness** | AI-generated summary: strong topics vs weak topics based on question-level performance across exams. | ✓ | ✓ |

---

## 4. Stoody API Integration Points

### 4.1 ExamPen Consumes from Stoody

| Stoody API | ExamPen Consumer | Trigger |
|---|---|---|
| `GET /api/students?class_id=&section_id=` | `svc-exam-orch` (fetch roster for exam) | Exam creation |
| `GET /api/tutors?subject_id=` | `svc-exam-orch` (assign invigilators/evaluators) | Exam creation |
| `GET /api/classes` | `svc-exam-orch` | Exam creation |
| `GET /api/subjects` | `svc-exam-orch` | Exam creation |
| `GET /api/users/{user_id}` | `svc-auth` (enrich JWT claims with Stoody profile) | Login |
| `GET /.well-known/jwks.json` | `svc-auth` (fetch signing keys for JWT validation) | Startup, cache refresh, `kid` mismatch |
| `GET /api/parents/{user_id}/children` | `svc-auth` (resolve parent access scope) | Parent access checks |

### 4.2 ExamPen Pushes to Stoody

| ExamPen Event | Stoody Webhook | Payload |
|---|---|---|
| `score.published` | `POST /api/webhooks/exampen/scores` | `{exam_id, scores: [{student_id, total, percentage, percentile}]}` |
| `exam.created` | `POST /api/webhooks/exampen/exams` | `{exam_id, subject_id, class_id, date, duration}` |
| `exam.completed` | `POST /api/webhooks/exampen/exams` | `{exam_id, status: 'completed', pens_synced, upload_status}` |

### 4.3 Embedding Strategy — DECIDED

**Decision: Option B — API-driven native embed.** Frozen. Not revisitable without architecture review.

Stoody's frontend calls ExamPen BFF APIs directly and renders ExamPen data in Stoody's own UI components. ExamPen does not serve its own HTML to Stoody users.

**Rationale:** iframe embed (Option A) was rejected because it fragments auth propagation (token passing via URL param or postMessage is fragile), breaks theming (ExamPen styles clash with Stoody shell), doubles observability surface (errors in iframe invisible to Stoody's error tracking), and creates a support boundary the user can feel.

**Implications:**

| Concern | How Option B Handles It |
|---|---|
| Auth propagation | Stoody frontend includes JWT in Authorization header on all ExamPen BFF calls. `svc-auth` validates via Stoody JWKS. |
| Routing | ExamPen screens are routes within Stoody's SPA. Stoody's router owns navigation. |
| Theming | Stoody's component library renders ExamPen data. ExamPen has no styling opinion. |
| Observability | All API errors surface in Stoody's error tracking. No hidden iframe failures. |
| API surface | ExamPen BFFs must expose comprehensive REST API. OpenAPI specs are the contract. |
| Dev coupling | Stoody frontend team builds ExamPen screens. ExamPen team provides BFF APIs + OpenAPI specs + mock servers. |

---

## 5. Mobile App — Stoody Integration

### 5.1 Tutor Mobile App

```
Stoody Tutor App
├── (Existing Stoody features: class management, content, etc.)
└── ExamPen Tab
    ├── Hub Control (Invigilator Mode)  ← BLE-dependent, works offline
    │   ├── Connect to hub
    │   ├── Register pens
    │   ├── Start/stop exam
    │   ├── Monitor sync
    │   ├── Upload data
    │   ├── Manual pen register
    │   └── Camera capture (copy images)
    ├── Score Management                 ← Cloud, works anywhere
    │   ├── Class score overview
    │   ├── Student drill-down
    │   ├── Score edit
    │   ├── Step-level marking
    │   ├── Feedback entry
    │   ├── Miss indicator review
    │   └── Plagiarism review
    ├── Objections                       ← Cloud
    │   ├── Inbox
    │   ├── Review + approve/reject
    │   └── Chat with student
    └── Analytics                        ← Cloud
        ├── Leaderboard
        └── Class stats
```

### 5.2 Student Mobile App

```
Stoody Student App
├── (Existing Stoody features: content, assignments, etc.)
└── ExamPen Tab
    ├── Upcoming Exams
    ├── Score View
    │   ├── Summary
    │   ├── Question breakdown
    │   ├── Answer image viewer
    │   ├── AI analysis (read-only)
    │   ├── Feedback view
    │   └── Miss indicators
    ├── Objections
    │   ├── File objection
    │   ├── Track status
    │   └── Chat with tutor
    └── Performance
        ├── Score history
        ├── Trend charts
        └── Strength/weakness analysis
```

---

## 6. Access Control Matrix

| Action | Super Admin | Principal | HOD | Tutor (Evaluator) | Tutor (Invigilator) | Student | Parent |
|---|---|---|---|---|---|---|---|
| Create exam | ✓ | ✓ | ✓ | ✓ (own subjects) | ✗ | ✗ | ✗ |
| Define rubric | ✓ | ✓ | ✓ | ✓ (own exams) | ✗ | ✗ | ✗ |
| Assign invigilators | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Start/stop exam (hub) | ✗ | ✗ | ✗ | ✗ | ✓ (assigned) | ✗ | ✗ |
| View all scores | ✓ | ✓ | ✓ (own dept) | ✓ (own exams) | ✗ | ✗ | ✗ |
| Edit scores | ✗ | ✗ | ✓ | ✓ (assigned as evaluator) | ✗ | ✗ | ✗ |
| Finalize scores | ✓ | ✓ | ✓ | ✓ (own exams) | ✗ | ✗ | ✗ |
| Review objections | ✗ | ✗ | ✓ | ✓ (assigned) | ✗ | ✗ | ✗ |
| View own scores | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ (child's) |
| File objection | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Chat (tutor side) | ✗ | ✗ | ✗ | ✓ (own students) | ✗ | ✗ | ✗ |
| Chat (student side) | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| View leaderboard | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ (own position) | ✓ (child's) |
| Export data | ✓ | ✓ | ✓ (own dept) | ✓ (own exams) | ✗ | ✗ | ✗ |
| Plagiarism review | ✓ | ✓ | ✓ | ✓ (own exams) | ✗ | ✗ | ✗ |

---

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-18 | Froze Stoody-issued JWT handling, added parent relationship integration, and replaced per-request token validation with JWKS-based validation flow. | Codex |
