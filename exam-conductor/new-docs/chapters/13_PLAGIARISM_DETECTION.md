# Chapter 13: Plagiarism Detection

## Status
- **Build status:** DRAFT
- **Authority source:** `api/plagiarism.openapi.yaml`

## Overview

Plagiarism detection is downstream of DCR and PCR evaluation. It consumes scoreable outputs and supporting evidence; it does not own ingest, OCR, or engine scoring.

```text
canonical artifacts -> DCR/PCR results -> plagiarism checks -> reviewable flags
```

## Alignment Rules

1. Plagiarism signals are advisory until reviewed.
2. Plagiarism logic reads engine outputs; it does not mutate raw canonical artifacts.
3. A plagiarism flag must be traceable back to the engine result or evidence set that produced it.

## Related Docs

- `api/plagiarism.openapi.yaml`
- `contracts/events/plagiarism.check.schema.json`
- `contracts/events/plagiarism.result.schema.json`
