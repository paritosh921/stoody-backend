# Chapter 24: Performance & Scaling

## Status
- **Phase:** Cross-cutting
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Performance characteristics and scaling strategy. Covers stroke ingestion
throughput (10K students burst), NATS JetStream backpressure handling,
TimescaleDB write optimization, AI pipeline batching, horizontal scaling of
stateless services, and hub capacity limits (40 pens per hub).

## Architecture Context

<!-- TODO: Scaling topology diagram showing stateless services (horizontal),
     stateful services (vertical + partitioning), and hub limitations.
     Reference Chapter 01. -->

## Detailed Design

### Stroke Ingestion Throughput
<!-- TODO: 10K students x 336 KB = 3.3 GB burst sizing, NATS buffering. -->

### Database Performance
<!-- TODO: TimescaleDB hypertable partitioning, PostgreSQL connection pooling,
     query optimization. -->

### AI Pipeline Batching
<!-- TODO: ONNX Runtime batch inference, GPU vs CPU tradeoffs. -->

### Horizontal Scaling
<!-- TODO: Stateless services behind load balancer, NATS consumer groups. -->

### Hub Capacity
<!-- TODO: 5 dongles x 8 pens = 40 max, multi-hub limitation (UR3). -->

### Cost Model
<!-- TODO: Target <Rs.2000/student/year, component cost breakdown. -->

## Interfaces
<!-- TODO: Rate limiting configuration, connection pool sizes. -->

## Configuration
<!-- TODO: NATS stream sizing, TimescaleDB chunk intervals, batch sizes. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md entries A8.4 (peak load),
     A8.8 (cost), UR3 (multi-hub). -->

## Testing
<!-- TODO: Reference test IDs E2E-08 (40-student full simulation),
     load testing approach. -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
