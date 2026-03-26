# Tamper Proof Spec

**Status:** ACTIVE  
**Authority:** Canonical artifact integrity, server-side fetch rules, and audit requirements.

---

## 1. Summary

Conducted-exam evaluation must be derived from canonical server-side artifacts. Clients must not be able to substitute corrected answer text into DCR or PCR evaluation flows.

This spec applies to:

- DCR conducted exams
- PCR conducted exams

It does not redesign practice persistence.

---

## 2. Integrity Model

```text
Student writes answer
      │
      ▼
┌─────────────────────────────────────────────┐
│  LAYER 1: Canonical Artifact Immutability   │
│  raw strokes / raw images -> content hash   │
│  write-once records in tenant MongoDB       │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│  LAYER 2: Server-Side Evaluation Fetch      │
│  engines fetch stored artifacts or text     │
│  client body never overrides canonical text │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│  LAYER 3: Append-Only Audit Trail           │
│  eval calls, overrides, flag resolutions    │
│  are preserved as history                   │
└─────────────────────────────────────────────┘
```

---

## 3. Layer 1: Canonical Artifact Immutability

Conducted-exam submissions are stored with a content hash and write-once semantics.

Required rules:

- raw strokes or images receive a SHA-256 `content_hash`
- canonical artifact documents carry `_immutable = true`
- repositories reject updates against immutable raw artifacts
- normalization and evaluation outputs are stored separately from raw artifact records
- object storage, where used, must preserve original versions

Applies to:

- `exampen_dcr_submissions`
- `evalpen_submissions`
- camera-originated conducted-exam assets

---

## 4. Layer 2: Server-Side Evaluation Fetch

Evaluation APIs identify server-side artifacts or response records. They must not accept student answer text as authoritative input for conducted exams.

Required request shape:

- DCR: authoritative request identifies submission or response records
- PCR: authoritative request identifies `response_id` and `question_id`

Required engine behavior:

- fetch canonical artifacts internally
- fetch `detected_text` from server-side records
- ignore any client-supplied corrected answer text in conducted-exam paths

---

## 5. Layer 3: Append-Only Audit Trail

Audit history must exist for:

- evaluation triggers
- gate calls and prompt lineage
- score overrides
- flag resolutions
- manual corrections

Minimum captured fields:

- `actor_id`
- `timestamp`
- `action`
- `before`
- `after`
- `reason`

For gate-linked evaluations, audit records should retain model and gate call references.

---

## 6. Correction and Override Rules

1. Raw canonical artifacts are never overwritten.
2. Detected text is not silently replaced.
3. Review actions append new records rather than mutating history away.
4. Score overrides require an actor and reason.
5. Flag resolution records must preserve before and after state.

---

## 7. Scope Boundary

### Included

- DCR conducted-exam evaluation
- PCR conducted-exam evaluation
- gate-linked audit references
- review and override auditability

### Excluded

- practice persistence redesign
- hub transport protocols
- Stoody frontend UX

Practice mode remains stateless from ExamPen's perspective and does not create new immutable practice artifacts inside ExamPen.
