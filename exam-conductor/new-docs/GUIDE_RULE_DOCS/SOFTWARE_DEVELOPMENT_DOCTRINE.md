# Software Development Doctrine

This document is a reusable engineering doctrine for planning, implementing, validating, and evolving software without drifting into a buggy patch cycle. It is written to be copied into other repositories and used as a standing rulebook for both humans and AI agents.

The central idea is simple:

- every important state must have one clear writable owner
- read operations must not secretly mutate durable state
- page/view/step transitions must be transactional when data can race
- tests must prove real behavior, not just produce green output

This doctrine applies to frontend, backend, desktop, mobile, data, infrastructure, and AI-assisted development.

Companion documents:

- `SYSTEM_DESIGN_GUIDELINE.md` for structured system-design questioning
- `SYSTEM_DESIGN_TEMPLATE.md` for per-feature or per-subsystem design capture
- `FEATURE_PLANNING_CHECKLIST.md` for compact start-of-work checks

## 1. Purpose

Use this doctrine to prevent:

- features that work locally but corrupt data under timing or retry conditions
- overlapping ownership of the same state across multiple layers
- hidden side effects in `load`, `fetch`, `hydrate`, `sync`, or `filter` functions
- false confidence from fake or incomplete tests
- AI-generated patches that "seem right" but do not verify the real behavior

The goal is not to stop change. The goal is to make change safe, local, and provable.

## 2. Mandatory Start-of-Work Questions

Every feature, bugfix, or refactor must start by answering these questions in order.

If any answer is unknown, work is not ready to implement.

### A. Problem and Intent

1. What user or system problem is being solved?
2. What is the current incorrect behavior?
3. What exact behavior will count as success?
4. Who is the audience for the change?
5. What is explicitly out of scope?

### B. Ownership and State

6. What state changes because of this work?
7. Which component, service, module, or model is the single writable owner of that state?
8. Which other layers may only read, derive, cache, or project that state?
9. Does any existing code already behave like a second owner?
10. If yes, how will that conflict be removed or constrained?

### C. Read and Write Boundaries

11. Which functions are read-only?
12. Which functions are allowed to mutate durable state?
13. Does any function name hide a side effect?
14. If a function is called `load`, `fetch`, `hydrate`, or `filter`, can it mutate local cache, database, or UI state?
15. If yes, should that be split into a pure read plus an explicit write step?

### D. Flow and Timing

16. What events can happen concurrently?
17. What can arrive late, retry, duplicate, or race?
18. What transitions must be transactional?
19. What happens if the user navigates away mid-operation?
20. What happens if stale data arrives after newer local changes?

### E. Failure Modes

21. What are the top 3 realistic failure modes?
22. What data can be lost, duplicated, or silently replaced?
23. What is the rollback or recovery behavior?
24. What remains safe under bad network, retry churn, or partial failure?
25. What will be logged or surfaced if the system cannot guarantee correctness?

### F. Test and Validation

26. What behavior-level tests prove the intended change?
27. What regression tests prove old correct behavior still works?
28. What edge-case tests cover race conditions and stale data?
29. What evidence is not enough to claim success?
30. What remains unverified even after implementation?

## 3. Development Stages and Exit Criteria

Work should move through the following stages. Skipping stages is one of the main causes of buggy development.

### Stage 1: Discovery

Goal:
- understand the real current behavior

Required work:
- trace the actual flow
- identify writable layers
- identify boundaries, queues, retries, caches, and persistence paths
- inspect existing tests and runtime contracts

Exit criteria:
- current flow is documented
- unknowns are reduced to real product decisions, not missing code facts

### Stage 2: Intent Locking

Goal:
- define what is being changed and what is not

Required work:
- success criteria
- audience
- in-scope / out-of-scope
- compatibility expectations

Exit criteria:
- a reviewer can tell whether the change succeeded without guessing

### Stage 3: Ownership and Invariants

Goal:
- decide who owns what

Required work:
- identify single writable owner for each critical state
- define invariants such as:
  - "only X mutates this state"
  - "Y only derives from X"
  - "hydration may merge but not replace dirty local state"
  - "page switch is transactional"

Exit criteria:
- ownership map and invariants are explicit and reviewable

### Stage 4: Interface and Failure Design

Goal:
- make the implementation safe before coding

Required work:
- define APIs, events, models, and side effects
- define merge rules and precedence
- define transactional boundaries
- define failure handling and fallback behavior

Exit criteria:
- implementer does not need to invent semantics during coding

### Stage 5: Implementation

Goal:
- change the smallest possible surface while preserving the chosen invariants

Required work:
- modify only the owners and adapters necessary for the change
- avoid introducing new overlapping owners
- keep read paths pure where possible

Exit criteria:
- code matches the planned ownership model

### Stage 6: Validation

Goal:
- prove the behavior, not just the code shape

Required work:
- run targeted behavior tests
- run regression tests
- run build/type checks
- verify logs, state transitions, and edge behavior where needed

Exit criteria:
- evidence exists for every major success claim

### Stage 7: Stabilization

Goal:
- decide whether the subsystem is actually healthy

Required work:
- document what was fixed
- list remaining risks
- decide whether the subsystem is now stable enough for more features

Exit criteria:
- either:
  - subsystem is stable enough for further work
  - or ownership must be simplified before more features are added

## 4. Core Engineering Rules

### Rule 1: One Writable Owner Per Critical State

If the same critical state can be mutated from multiple places, bugs will accumulate.

Examples:
- a shopping cart should not be independently mutated by UI local state, cache, websocket reducer, and server hydration logic
- a live document should not have separate authoritative versions in editor state, autosave state, and collaboration state

Correct pattern:
- one owner mutates
- all other layers read, derive, cache, or project

### Rule 2: Reads Must Be Pure by Default

A function named `load`, `fetch`, `hydrate`, `read`, `filter`, or `list` must not silently mutate durable state unless its contract explicitly says so.

Bad:
- `loadFromServer()` updates local cache, replaces local state, and triggers UI changes

Good:
- `fetchRemotePage()` returns data
- `mergeRemoteIntoLocal()` performs the explicit write

### Rule 3: Transactions for Boundary Events

Any boundary that can race must be treated as a transaction.

Typical boundaries:
- page switch
- tab switch
- route change
- close/unmount
- submit
- retry / reconnect
- offline-to-online transition

Transactional sequence:
1. drain pending work
2. finalize current state
3. persist durable snapshot
4. mark sync work
5. switch identity
6. load next state

### Rule 4: Normalize at Ingress

Normalize incoming data as early as possible.

Examples:
- normalize IDs, types, timestamps, scopes, and ownership keys at the boundary
- do not let noisy metadata flow deep into merge logic

This prevents later helpers from becoming destructive filters.

### Rule 5: Derived Views Must Not Become Hidden Owners

UI state, cache state, and rendered state often start as "views" and accidentally become partial owners over time.

Any derived view that writes back must be reviewed as a potential new owner.

## 5. AI Agent Hard Boundaries

These rules are mandatory for AI-assisted development.

### Planning and Implementation

- Do not implement until the writable owner, invariants, and success criteria are explicit.
- Do not introduce a second writable owner for an existing critical state without documenting the decision.
- Do not bypass architectural uncertainty by "patching both places" unless the plan explicitly calls for a temporary compatibility layer.
- Do not hide side effects inside helper functions with misleading names.

### Testing and Validation

- Do not claim a fix is verified unless a real test or real runtime validation was executed.
- Do not count typecheck, lint, or build success as behavior validation.
- Do not count an empty test file, placeholder assertion, or mocked-away behavior as a real test.
- Do not weaken assertions only to get green tests unless the product spec explicitly changed.
- Do not remove or skip failing tests to make a patch appear successful.
- Do not report "all tests passed" if the relevant behavior was never exercised.
- Always state:
  - exactly what was tested
  - how it was tested
  - what remains untested

### Honesty Requirements

An AI agent must explicitly say one of the following:

- "Verified by behavior test"
- "Verified by build/typecheck only"
- "Not verified in runtime"
- "Could not test this path"

It must never collapse these into one success statement.

## 6. Test Integrity Rules

### A test only counts if it exercises the claim

Examples:

- Claim: "page switch no longer loses recent edits"
  - valid test: write, switch immediately, return, confirm edits remain
  - invalid test: component renders, build passes

- Claim: "server hydration no longer overwrites local dirty data"
  - valid test: create local dirty state, simulate stale server response, confirm local survives
  - invalid test: API returns 200

- Claim: "retry logic is safe"
  - valid test: induce reconnect churn or repeated delayed responses
  - invalid test: one normal success path

### Required test categories for stateful features

Every non-trivial stateful feature should have:

- happy-path test
- regression test for the bug being fixed
- stale/late/retry race test
- navigation or boundary-transition test
- persistence or recovery test if durable data is involved

### What is not enough

The following are not enough on their own:

- build passes
- typecheck passes
- linter passes
- snapshot updated
- mocked test passes without real state transition
- one manual click-through with no edge case coverage

## 7. Reusable Design Pattern: Single Writable Owner

This pattern can be used for editors, carts, forms, drafts, queues, multiplayer state, and sync-heavy UI.

### Generic model

```text
┌───────────────────────────────┐
│   ActiveDomainModel           │
│ - identity                    │
│ - primary data                │
│ - dirty flag                  │
│ - version / freshness         │
│ - boundary methods            │
├───────────────────────────────┤
│ submit(event)                 │
│ snapshot()                    │
│ merge(remote)                 │
│ switch(identity)              │
└───────────────────────────────┘
         │ derives
    ┌────┴────┐
    ▼         ▼
   UI       Cache/Sync adapters
```

Rules:
- only the model mutates primary state
- UI renders from the model
- cache stores snapshots from the model
- server sync submits snapshots from the model
- remote hydration merges into the model
- transitions are transactional

## 8. Common Failure Patterns

Watch for these recurring smells:

- a "load" function also saves
- a "filter" function drops valid data because metadata is incomplete or noisy
- multiple layers believe they own the same entity
- page/view identity changes before prior state is durably persisted
- retry/reconnect logic replays stale data over fresh local work
- UI state and persistent state drift because timing assumptions differ by one render, one event, or one retry

If these patterns appear, do not keep patching indefinitely. Re-evaluate ownership and boundaries.

## 9. Review Checklist

Before approving implementation, confirm:

- the problem and success criteria are explicit
- each critical state has one writable owner
- read functions are not hiding destructive writes
- transactional boundaries are identified
- failure modes were discussed
- tests actually exercise the claim
- untested risks are stated honestly

## 10. Working Rule for Teams

A team should assume:

- features will continue to be added and removed
- bugs are normal
- architecture drift is normal

The way out of a buggy development cycle is not to stop shipping. It is to enforce:

- explicit ownership
- small change surfaces
- honest validation
- repeated documentation of invariants
- periodic simplification before adding too much more complexity

When the same class of bug appears repeatedly, stop adding features on top of it and simplify the ownership model first.
