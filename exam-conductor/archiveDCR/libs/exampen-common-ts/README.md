# @exampen/common-ts

Shared TypeScript utilities for all ExamPen frontend and mobile projects.

## Install

This is a workspace-local package. Reference it via your monorepo tooling (npm workspaces, pnpm, turborepo, etc.):

```jsonc
// consumer package.json
{
  "dependencies": {
    "@exampen/common-ts": "workspace:*"
  }
}
```

## Usage

### Auth helpers

```ts
import {
  getAuthHeaders,
  parseJwtClaims,
  isTokenExpired,
  storeToken,
  getToken,
  clearToken,
} from '@exampen/common-ts';

// Store token after Stoody login
storeToken(stoodyJwt);

// Build headers for API calls
const headers = getAuthHeaders(stoodyJwt);
// => { Authorization: 'Bearer ...', 'X-Requested-With': 'ExamPen' }

// Decode claims (no signature verification — that is server-side)
const claims = parseJwtClaims(stoodyJwt);
console.log(claims.user_id, claims.exampen_roles);

// Check expiry with 30-second grace
if (isTokenExpired(stoodyJwt)) {
  clearToken();
}
```

### Typed API client

```ts
import { apiGet, apiPost, apiPatch } from '@exampen/common-ts';
import type { Exam, Score, ApiResponse } from '@exampen/common-ts';

// GET with auto-injected auth headers and 503 retry
const exams = await apiGet<{ items: Exam[] }>('/api/v1/teacher/exams');

// POST with a body
const score = await apiPost<Score>(
  '/api/v1/scores/abc-123/finalize',
  { actor_id: 'teacher-1' },
);

// PATCH with custom options
const updated = await apiPatch<Score>(
  '/api/v1/scores/abc-123/students/s-1/questions/q-1',
  { teacher_id: 't-1', new_score: 8, reason: 'Step 2 was correct' },
  { maxRetries: 3 },
);
```

### WebSocket (invigilator console)

```ts
import { connectWs } from '@exampen/common-ts';
import type { WebSocketEnvelope } from '@exampen/common-ts';

const ws = connectWs({
  onMessage: (envelope: WebSocketEnvelope) => {
    switch (envelope.event_type) {
      case 'sync.progress':
        // update pen sync UI
        break;
      case 'dongle.health':
        // update dongle status
        break;
    }
  },
  onError: (e) => console.error('WS error', e),
});

// Close when done
ws.close();
```

### Shared types

```ts
import type {
  ExamStatus,
  ScoreStatus,
  ObjectionStatus,
  MissIndicatorState,
  ExamPenRole,
  ChatMessage,
} from '@exampen/common-ts';

// All FSM state unions match the OpenAPI specs exactly
const status: ExamStatus = 'timer_running';
const scoreState: ScoreStatus = 'ai_draft';
```

## Build

```bash
npm install
npm run build     # emits to dist/
npm run typecheck  # type-check without emitting
```

## Design decisions

- **No runtime dependencies** — only TypeScript as a dev dependency.
- **Interfaces over classes** — all data shapes are plain interfaces.
- **String literal unions over enums** — FSM states use `type X = 'a' | 'b'` for tree-shaking and JSON compatibility.
- **No signature verification** — `parseJwtClaims` decodes the payload only. Verification is the responsibility of `svc-auth`.
- **localStorage for token storage** — gracefully no-ops in non-browser environments (SSR, Node).
