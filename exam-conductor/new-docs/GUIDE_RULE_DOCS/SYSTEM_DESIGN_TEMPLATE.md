# System Design Template

Use this template after applying `SYSTEM_DESIGN_GUIDELINE.md`.

Fill it for each subsystem, feature, or architectural change before implementation planning begins.

## 1. Problem Statement

- Feature / subsystem name:
- One-paragraph problem statement:
- Why this matters:
- Main actors / users:

## 2. Success Criteria

- User-visible success:
- System-level success:
- Must preserve:
- Out of scope:

## 3. Existing Context

- Existing components / services involved:
- Existing constraints:
- Existing compatibility requirements:
- Known debt or risks already present:

## 4. Critical State and Ownership

| State | Writable Owner | Readers / Derivers / Caches | Notes |
|---|---|---|---|
| | | | |

Required statement:

- The single writable owner of each critical state is:

## 5. Interfaces and Flow

### Inputs

- APIs:
- Events:
- Commands:
- User actions:

### Outputs

- Persisted data:
- UI updates:
- Background jobs:
- External side effects:

### Normal Flow

1.
2.
3.

### Boundary / Transactional Flow

1.
2.
3.

## 6. Read / Write Boundary Rules

- Read-only functions:
- Durable write functions:
- Caches:
- Source(s) of truth:
- Any risky side-effectful read paths:

## 7. Failure Modes

| Failure Mode | What Breaks | Mitigation | Residual Risk |
|---|---|---|---|
| | | | |

## 8. Non-Functional Requirements

- Latency expectations:
- Consistency model:
- Offline / degraded behavior:
- Concurrency considerations:
- Scale considerations:
- Security / privacy / audit notes:

## 9. Validation Plan

- Happy-path test:
- Regression test:
- Retry / stale / race test:
- Boundary-transition test:
- Persistence / recovery test:
- Observability / logs / metrics needed:

## 10. AI Agent Verification Notes

- What counts as real validation for this design:
- What would not count as evidence:
- What may remain unverified after first implementation:

## 11. Assumptions

- Assumption 1:
- Assumption 2:
- Assumption 3:

## 12. Ready-to-Implement Check

Implementation planning may start only if all are true:

- [ ] success criteria are explicit
- [ ] writable owners are identified
- [ ] interfaces are described
- [ ] transactional boundaries are identified
- [ ] failure modes are named
- [ ] validation is specific
- [ ] assumptions are recorded
