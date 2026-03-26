# Chapter 19: Infrastructure & Deployment

## Status
- **Build status:** DRAFT

## Overview

Infrastructure must support:

- shared ingest substrate
- independent DCR and PCR engines
- shared LLM gate
- tenant/admin MongoDB storage

## Current Alignment Rules

1. Do not document PostgreSQL as the active storage model for ExamPen.
2. Keep hub deployment concerns under `integration/HUB_DEPLOYMENT_SPEC.md`.
3. Treat this chapter as explanatory only.
