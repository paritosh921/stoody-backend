# Chapter 05: Authentication and Authorization

## Status
- **Build status:** DRAFT
- **Authority source:** `integration/STOODY_INTEGRATION_SPEC.md`

## Overview

Stoody remains the source of truth for identity, tutor visibility, and parent/student relationships.

ExamPen consumes Stoody identity and applies it to:

- conducted-exam access
- tutor/student views
- practice boundary enforcement

## Key Rules

1. Stoody owns user identity.
2. Tutor visibility follows the existing admin-owned student visibility model.
3. This chapter does not introduce a second ownership model for access.
4. Practice persistence remains in the existing backend path.

## Related Docs

- `integration/STOODY_INTEGRATION_SPEC.md`
- `governance/STATE_OWNERSHIP_MAP.md`
- `architecture/DUAL_MODE_ARCHITECTURE.md`
