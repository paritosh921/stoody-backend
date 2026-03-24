# Chapter 20: Monitoring & Observability

## Status
- **Phase:** P0
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Observability stack: Grafana for dashboards, Loki for log aggregation, Tempo for
distributed tracing, Prometheus for metrics. Covers service health monitoring,
NATS consumer lag alerting, pipeline throughput tracking, and hub fleet status.

## Architecture Context

<!-- TODO: Diagram showing Grafana + Loki + Tempo + Prometheus collecting from
     all services and hub-uplink. Reference Chapter 01. -->

## Detailed Design

### Metrics Collection
<!-- TODO: Prometheus scrape targets, custom metrics per service. -->

### Log Aggregation
<!-- TODO: Loki configuration, structured logging format, log retention. -->

### Distributed Tracing
<!-- TODO: Tempo configuration, trace context propagation across NATS events. -->

### Dashboards
<!-- TODO: Key Grafana dashboards: pipeline throughput, NATS lag, service health,
     hub fleet status. -->

### Alerting
<!-- TODO: Alert rules: NATS consumer lag >10s, service down, error rate spike,
     duplicate rate >1%. -->

## Interfaces
<!-- TODO: Grafana endpoints, Prometheus /metrics paths. -->

## Configuration
<!-- TODO: Grafana provisioning, Prometheus scrape config, Loki retention. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md entries A8.4 (peak load
     monitoring), A8.6 (duplicate rate monitoring). -->

## Testing
<!-- TODO: Reference observability integration test IDs if any. -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
