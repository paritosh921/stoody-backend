# Chapter 20: Monitoring and Observability

## Status
- **Build status:** DRAFT

## Overview

Observability must track the four active architecture domains separately:

- shared ingest substrate
- DCR engine
- PCR engine
- shared LLM gate

```text
ingest metrics   ─┐
DCR metrics      ├-> dashboards / alerts / traces
PCR metrics      ┤
gate metrics     ┘
```

## Key Signals

- ingest durability, backlog, duplicate rate
- DCR throughput, confidence distribution, fallback rate
- PCR segmentation flags, auto-eval rate, review backlog
- gate budget headroom, token usage, refusal counts

## Alignment Rules

1. Metrics should preserve the boundary between collection and evaluation.
2. Gate metrics must be visible independently of PCR metrics.
3. Practice metrics must not imply a new ExamPen practice persistence model.

## Related Docs

- `architecture/LLM_GATE_SPEC.md`
- `governance/FAILURE_MITIGATION_REGISTER.md`
- `governance/TEST_SUITE_SPEC.md`
