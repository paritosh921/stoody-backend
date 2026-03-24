# Feature Planning Checklist

Use this checklist at the start of every feature, bugfix, or refactor. It is the compact execution companion to `SOFTWARE_DEVELOPMENT_DOCTRINE.md`.

Do not begin implementation until every section below has a concrete answer.

## 1. Problem and Success

- What exact problem is being solved?
- What is the current wrong behavior?
- What is the expected correct behavior?
- Who is affected by this change?
- What is explicitly out of scope?

## 2. Ownership and State

- What state changes because of this work?
- Who is the single writable owner of that state?
- Which layers may only read, derive, cache, or project that state?
- Does any current code behave like a second owner?
- If yes, how will that conflict be prevented?

## 3. Interfaces and Boundaries

- Which functions are read-only?
- Which functions are allowed to mutate durable state?
- Does any `load`, `fetch`, `hydrate`, `sync`, or `filter` function have side effects?
- If yes, should it be split into a pure read plus explicit write step?
- What inputs, outputs, events, or schemas change?

## 4. Timing and Failure Modes

- What can race, retry, duplicate, or arrive late?
- What happens if the user navigates away mid-operation?
- What happens if stale remote data arrives after newer local data?
- What transition must be transactional?
- What are the top 3 realistic failure modes?

## 5. Validation Plan

- What happy-path test proves the feature works?
- What regression test proves the bug is fixed?
- What edge-case or race test is required?
- What persistence/recovery test is required, if data is involved?
- What evidence is not enough to claim success?

## 6. AI Agent Hard Boundaries

- Do not claim success from build/typecheck alone.
- Do not count empty, placeholder, or mocked-away tests as real validation.
- Do not weaken assertions just to get green tests unless the spec changed.
- Do not remove or skip failing tests to make a patch appear successful.
- Always state:
  - what was tested
  - how it was tested
  - what remains untested

## 7. Ready-to-Implement Gate

Implementation may start only when all of the following are true:

- the user-facing success criteria are explicit
- the writable owner is identified
- read/write boundaries are clear
- failure modes are named
- the validation plan is specific
- unverified risks are acknowledged

If any of these are missing, continue planning instead of coding.
