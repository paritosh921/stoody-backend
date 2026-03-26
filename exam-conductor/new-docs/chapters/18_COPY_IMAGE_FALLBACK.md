# Chapter 18: Copy Image Fallback

## Status
- **Build status:** DRAFT
- **Authority source:** `api/copy-upload.openapi.yaml`

## Overview

Copy-image fallback is a camera-based conducted-exam ingest path. It feeds the shared ingest substrate and is primarily routed to PCR because PCR can normalize image pages into `PageOCR`.

```text
camera upload -> canonical image artifact -> exam_type / source routing -> PCR engine
```

## Alignment Rules

1. Camera fallback does not redefine hub ownership of pen-originated artifacts.
2. Image uploads must preserve the same provenance fields as pen uploads.
3. Fallback images do not bypass tamper-proof or review rules.

## Related Docs

- `api/copy-upload.openapi.yaml`
- `architecture/PCR_EVAL_ENGINE_SPEC.md`
- `architecture/TAMPER_PROOF_SPEC.md`
