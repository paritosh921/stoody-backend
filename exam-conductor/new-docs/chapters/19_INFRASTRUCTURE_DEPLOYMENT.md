# Chapter 19: Infrastructure & Deployment

## Status
- **Phase:** P0 / P2a
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Infrastructure stack: Docker Compose for local dev and staging, Traefik as API
gateway, PostgreSQL 16 with TimescaleDB, NATS JetStream for events, MinIO for
object storage. Hub deployment via golden image. Server deployment via container
orchestration.

## Architecture Context

<!-- TODO: Infrastructure topology diagram. Reference Chapter 01. -->

## Detailed Design

### Docker Compose Stack
<!-- TODO: Service definitions, networking, volume mounts, health checks. -->

### Database Infrastructure
<!-- TODO: PostgreSQL per-service schemas, RLS setup, TimescaleDB hypertables,
     pgBackRest backup. -->

### Message Broker
<!-- TODO: NATS JetStream configuration, stream/consumer definitions. -->

### Object Storage
<!-- TODO: MinIO buckets, cross-region replication, lifecycle policies. -->

### API Gateway
<!-- TODO: Traefik routing, rate limiting, TLS termination. -->

### Hub Golden Image
<!-- TODO: RPi image build process, Ubuntu Server 24.04 LTS arm64,
     systemd service definitions. Reference HUB_DEPLOYMENT_SPEC.md. -->

## Interfaces
<!-- TODO: Exposed ports, DNS entries, environment variable catalog. -->

## Configuration
<!-- TODO: docker-compose.yml structure, .env files, secrets management. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md entries A8.1 (multi-tenant
     leak), A8.5 (backup failure), A8.8 (cost). -->

## Testing
<!-- TODO: Reference infra-related test IDs, L1 (build verified). -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
