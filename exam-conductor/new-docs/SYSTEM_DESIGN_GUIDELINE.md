# System Design Guideline

This document is a reusable system design guide for humans and AI agents. It is intended to be used before implementation planning begins.

Use it when:

- designing a new subsystem
- extending an existing system with non-trivial state or data flow
- changing boundaries between frontend, backend, worker, agent, cache, queue, or database layers
- adding async, offline, sync, retry, or real-time behavior

This guide is meant to force the right questions early so that implementation does not miss critical design decisions.

## 1. What System Design Is

System design is the discipline of deciding:

- what the major parts of the system are
- how data moves through them
- who owns what state
- where mutation is allowed
- what happens under failure, retries, scale, stale data, and concurrency

System design is not just architecture drawing. It is a decision process that must end with:

- explicit ownership
- explicit boundaries
- explicit contracts
- explicit failure behavior
- explicit validation strategy

## 2. How This Guide Must Be Used

Before asking the user questions, an AI agent must first discover what it can from the repo, docs, types, existing APIs, config, and current architecture.

After that, the agent must move through the phases below in order.

The agent must not jump to implementation planning until the design is decision complete.

## 3. Phase 1: Requirements Interrogation

The agent must establish:

- What problem is being solved?
- Who are the actors and users?
- What exact behavior is expected?
- What is in scope and out of scope?
- What constraints already exist?

Questions to answer:

1. What is the problem statement in one paragraph?
2. What is the user-visible success criterion?
3. What are the main actors, systems, or roles involved?
4. What existing behavior must be preserved?
5. What is explicitly out of scope for this change?
6. What constraints are already fixed by product, compliance, platform, or compatibility?

Exit criteria:

- the problem and success criteria can be stated without ambiguity

## 4. Phase 2: System Boundary and Ownership Interrogation

The agent must identify all critical state and assign one writable owner per important state.

Questions to answer:

1. What state exists in this system?
2. Which state is critical to correctness?
3. Who is the single writable owner of each critical state?
4. Which other layers are allowed only to read, derive, cache, or project it?
5. Does any existing code already behave like a second owner?
6. Which transitions must be transactional?

Required outputs:

- state inventory
- owner map
- source-of-truth map
- cache/projection map

Hard rule:

- if one critical state has multiple writable owners, the design is not complete

## 5. Phase 3: Interface and Flow Interrogation

The agent must define the communication model.

Questions to answer:

1. What are the inputs and outputs?
2. What APIs, events, commands, queues, or callbacks are involved?
3. Which operations are reads and which are writes?
4. Does any read path have a write side effect?
5. What sequence of steps occurs in the normal path?
6. What sequence of steps occurs during retries, reconnects, or partial failure?

Required outputs:

- data flow diagram or narrative
- interface list
- read/write boundary list

Hard rule:

- functions named `load`, `fetch`, `read`, `hydrate`, or `filter` must not silently mutate durable state unless the contract explicitly says so

## 6. Phase 4: Non-Functional Interrogation

The agent must not design only for the happy path.

Questions to answer:

1. What are the latency expectations?
2. What concurrency or contention exists?
3. What scale matters now and later?
4. What happens offline or under degraded network?
5. What consistency level is required?
6. What security, privacy, or audit constraints exist?

Use this section to distinguish:

- what must be strongly consistent
- what can be eventual
- what must never be lost
- what can be recomputed

## 7. Phase 5: Failure-Mode Interrogation

The agent must walk through realistic failure paths.

Questions to answer:

1. What can arrive late, duplicate, retry, or race?
2. What happens if the user navigates away mid-operation?
3. What happens if stale remote data arrives after newer local data?
4. What happens if two layers disagree on ownership?
5. What data can be lost, duplicated, or silently replaced?
6. What is the rollback, repair, or recovery path?

Required outputs:

- top 3 to 5 failure modes
- mitigation per failure mode
- unresolved risk list

Hard rule:

- if the design cannot explain how stale, duplicate, or delayed data is handled, it is not design complete

## 8. Phase 6: Validation Interrogation

The agent must define how the system design will be proven, not just implemented.

Questions to answer:

1. What test proves the happy path?
2. What test proves the most likely regression does not happen?
3. What test proves boundary transitions are safe?
4. What test proves stale/retry/race behavior is safe?
5. What observability or logging is required?
6. What remains unverified after initial delivery?

Required outputs:

- validation plan
- regression plan
- observability notes

## 9. AI Agent Questioning Protocol

When using this guide interactively, an AI agent should:

1. Explore first for discoverable facts.
2. Ask questions only for:
   - preferences
   - missing requirements
   - missing constraints
   - unresolved ownership or boundary decisions
3. Ask in phases, not randomly.
4. Keep track of:
   - known facts
   - assumptions
   - user choices
   - unresolved risks
5. Refuse to move to implementation planning until:
   - ownership is explicit
   - major failure modes are named
   - validation is specified

The agent should ask questions in this order:

1. Requirements
2. Ownership
3. Interfaces
4. Failure behavior
5. Validation

The agent should not start with implementation details unless the design questions are already resolved.

## 10. Conditional Branches for Different System Types

After the core phased checklist, the agent should branch based on the system type.

### A. Stateful Local-First Systems

Ask:

- What is the single in-memory owner?
- What is durable local cache vs source of truth?
- Can remote hydration merge but never replace dirty local state?
- What events must be transactional?

### B. Distributed or Multi-Service Systems

Ask:

- What service owns the data?
- What consistency is required across services?
- What is the retry and idempotency model?
- What happens when one service is ahead of another?

### C. Async/Event-Driven Systems

Ask:

- What events can duplicate or reorder?
- How is idempotency enforced?
- What happens if consumers are slow or unavailable?
- How is replay handled?

### D. CRUD-Heavy Systems

Ask:

- What validation belongs in client vs server?
- What constraints must be enforced centrally?
- What fields are derived vs user-controlled?
- What are migration implications?

### E. Real-Time or Collaborative Systems

Ask:

- What is local immediate truth vs remote shared truth?
- How are conflicts resolved?
- What is authoritative during reconnect churn?
- What visible behavior is acceptable under temporary disagreement?

## 11. Design Completion Gate

The design is ready for implementation planning only when all of the following are true:

- problem and success criteria are explicit
- critical state owners are identified
- read/write boundaries are explicit
- transactional boundaries are defined
- top failure modes are addressed
- validation is specific
- unresolved risks are recorded honestly

If any of these are missing, stay in design mode.

## 12. What a Good Final Design Output Must Contain

A final design summary should contain:

- problem statement
- success criteria
- actors and scope
- critical state ownership
- interfaces and data flow
- failure mode handling
- validation strategy
- explicit assumptions

If any of these are absent, the design is incomplete.
