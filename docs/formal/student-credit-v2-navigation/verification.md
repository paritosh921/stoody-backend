# Formal verification record

Date: 2026-08-12  
Status: formal gate rechecked after implementation; confirmed-as-built closeout complete

TLC exit status: 0
States generated: 670913
Distinct states: 161424

TLC checked the bounded abstract model, not the implementation. Application behavior is established separately by the tests and builds recorded below and in `implementation-evidence.md`.

## Translation

Command:

```powershell
D:\SOFTWARE_Projects_LP\skiller-bot\backend\venv\Scripts\python.exe C:\Users\ashue\.agents\skills\pluscal\scripts\tla_tool.py translate Model.tla
```

Result: PlusCal parsing and translation completed successfully.

## Counterexamples and design corrections

1. The first liveness run allowed the mobile process to run forever while a credit source actor was ignored. The model was corrected to apply fairness per worker/admin/UI process, making the assumption of eventual worker availability explicit.
2. The next run showed that a one-time UTC rollover was insufficient: a later legacy job could again make current-day awards exceed the V2 cap before activation. The day-boundary abstraction was made recurring, and the operational activation guard now explicitly refuses while any student is above the V2 daily cap.
3. A larger combined liveness state space exceeded the initial command timeout. Verification was split into the usual bounded configurations: one active source with temporal properties, and two active sources with all safety invariants. This changes exploration size, not the modeled transitions.

## Liveness configuration

Command:

```powershell
D:\SOFTWARE_Projects_LP\skiller-bot\backend\venv\Scripts\python.exe C:\Users\ashue\.agents\skills\pluscal\scripts\tla_tool.py check Model.tla --config Model.cfg
```

Result:

- No error found.
- 14,449 states generated.
- 4,104 distinct states.
- State graph depth 20.
- Zero states left on queue.
- Checked `AllJobsEventuallyTerminal`, `ActivationEventuallySucceeds`, and `RequestedNavigationCommits` plus all configured invariants.

## Two-source safety configuration

Command:

```powershell
D:\SOFTWARE_Projects_LP\skiller-bot\backend\venv\Scripts\python.exe C:\Users\ashue\.agents\skills\pluscal\scripts\tla_tool.py check Model.tla --config ModelSafety.cfg
```

Result:

- No error found.
- 670,913 states generated.
- 161,424 distinct states.
- State graph depth 30.
- Zero states left on queue.
- Checked type/coherent snapshot/pinning, strict caps, ledger/award consistency, bounded unrecoverable completion, activation, UI identity-only/menu shape, award formula, and tier-boundary invariants.

## Formal-to-code obligations

- Implement a tenant policy-transition lock around enqueue snapshot/insert and V2 activation.
- Refuse V2 activation on open jobs or any student above 100 current UTC-day positive awards.
- Add semantic policy validation and API conflict/validation responses.
- Bound missing-completion reconciliation and persist a stable terminal reason.
- Return the complete tier ladder in summary projections; remove client threshold calculations.
- Keep immutable photo submission version semantics and same-evidence idempotency.
- Make the avatar overlay identity-only and mutually exclusive with the hamburger.
- Put Credits and Rewards adjacent in both role menus; remove the student Profile credits projection.
- Ensure Rewards has no mutation path.

## Separate-validation reminder

TLC does not establish MongoDB/Celery/React Native/API behavior. Focused backend tests, mobile Jest/type checks, frontend tests/build, and Android build remain required after implementation.
