# Chapter 23: Testing Strategy

## Status
- **Build status:** DRAFT
- **Authority source:** `governance/TEST_SUITE_SPEC.md`

## Overview

Testing follows the active architecture boundaries instead of the older single-pipeline model.

```text
L3 unit        -> isolated ingest / DCR / PCR / gate logic
L4 integration -> storage, API, routing, review, usage flows
L5 end-to-end  -> conducted exam and practice paths
L6/L7          -> hub hardware and field validation
```

## Required Coverage

- shared ingest substrate durability and routing
- DCR deterministic scoring and optional fallback behavior
- PCR segmentation, flagging, evaluation, and practice endpoint behavior
- shared gate budget and logging behavior
- Stoody integration and visibility boundaries

## Alignment Rules

1. Use the IDs in `governance/TEST_SUITE_SPEC.md`.
2. No test plan should assume PostgreSQL or a monolithic evaluator.
3. Practice tests must preserve the unchanged persistence boundary.

## Related Docs

- `governance/TEST_SUITE_SPEC.md`
- `governance/FAILURE_MITIGATION_REGISTER.md`
- `architecture/unifiedPlan.md`
