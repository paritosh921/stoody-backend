# Practice answer persistence and submission

## Contract

- An attempt is resolved on the server by authenticated user and practice document. The browser's previous attempt ID is adopted on first migration. A device with pending ink resumes that attempt rather than relabelling its strokes.
- Question ledgers use `(attempt ID, question ID)`, not a question's position in the list. Page identity is `(copy ID, book type, physical page number)`.
- Browser writing is journaled in IndexedDB with tenant/user isolation. Authentication cleanup does not delete writing. Upload acknowledgement clears a pending revision only if no newer writing has arrived. localStorage is a disposable rendering cache.
- Browser and agent page uploads use the same additive merge and version compare-and-swap. Existing geometry is immutable by stroke ID. A duplicate can fill missing ownership; contradictory ownership is rejected. Replays cannot erase ownership.
- `POST /practice/answers/prepare` discovers exactly owned server strokes, verifies every expected local stroke ID, and freezes rendered evidence and its receipt in `practice_answer_snapshots`. It does not require all answer pages to be cached locally.
- `POST /practice/evaluate` with `snapshotId` uses that immutable evidence. The snapshot is checked against authenticated user, attempt, and question. Text/attachment changes produce a separate evaluation key. Concurrent identical evaluations are rejected; completed retries return the stored result. A 15-minute lease allows recovery after a crashed worker.
- Evaluation result and history payload are committed together. History projection is idempotent and is repaired on retry without another LLM call.

## Legacy recovery

Old numeric page references are candidates, not ownership. Server recovery searches their original book identities and prefers explicit attempt/question tags. Legacy untagged strokes require the stored disjoint time windows. An existing candidate with no matching evidence is excluded; a missing candidate or multiple matching books reports a reconciliation error. Missing expected stroke IDs always block incomplete submission. Unsupported/noncanonical writing is not silently converted to trusted evidence.

No production data deletion or bulk ownership reassignment is required. New collections use deterministic `_id` keys and MongoDB's existing unique `_id` indexes. The existing unique canvas page index must include user, copy, book, and page.

## Deployment and acceptance

Deploy the backend before the frontend. The backend retains the old evaluation request contract for older web/mobile clients. Deploying the frontend first causes its new draft/prepare requests to fail visibly.

Automated checks:

```text
Backend: python -m pytest tests/test_practice_answer_lifecycle.py tests/test_practice_canvas_contract.py tests/test_canvas_stroke_canonical_contract.py tests/test_practice_language_feedback.py tests/test_practice_latex_normalization.py -q
Frontend: npm run test:practice-session
Frontend: npm run build
```

Physical-pen acceptance after deployment: write across two pages, submit, log out/in, reopen without visiting the older page, and submit again. Repeat with an offline interval, a reconnect, and identical page numbers in different books. Verify that all expected stroke IDs reach the snapshot, no foreign question ink enters it, and a completed retry returns the same evaluation. Browser cache eviction must not change the answer.

Truly missing/ambiguous historical ink still requires reconciliation. Network/storage failures remain visible and preserve pending work; this contract does not claim hardware or network failures are impossible. A real pen and deployed end-to-end run are separate acceptance steps from the automated tests.
