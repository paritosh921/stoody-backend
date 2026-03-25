# STOODY_INTEGRATION_SPEC.md
# ExamPen × Stoody Integration

Reference: `architecture/DUAL_MODE_ARCHITECTURE.md`, `architecture/PCR_EVAL_ENGINE_SPEC.md`

---

## 1. Integration Summary

ExamPen remains a subsystem of Stoody. Stoody owns identity, roster, tutor/student relationships, and current practice persistence.

ExamPen adds:

- conducted-exam ingest substrate
- DCR engine
- PCR engine
- shared LLM gate

```text
Stoody identity / roster / tutor visibility
                 │
                 ▼
      ┌─────────────────────────────┐
      │         ExamPen             │
      │ ingest substrate + engines  │
      └─────────────────────────────┘
                 │
                 ▼
     Tutor/student views inside Stoody
```

---

## 2. Identity and Visibility

- Stoody remains the source of truth for users and roles.
- Tutor access to exam data follows the existing admin-owned student visibility model.
- Exam-conducted records are stored in tenant/admin MongoDB.
- Practice persistence remains in the existing Stoody backend path.

---

## 3. Practice Boundary

PCR may expose a live practice evaluation endpoint, but:

- it is stateless from ExamPen's perspective
- it does not create new ExamPen practice collections
- the current backend continues to persist practice results as it already does

---

## 4. Integration Rules

1. Do not redesign Stoody practice storage here.
2. Do not introduce a second ownership model for tutor visibility.
3. Conducted-exam data and practice data remain distinct.
