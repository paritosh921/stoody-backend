# hub-common

Shared IPC protocol, message type catalog, and configuration loader for all ExamPen hub modules.

## Contents

| File | Purpose |
|------|---------|
| `hub_common/ipc_protocol.py` | `IpcEnvelope` dataclass, `IpcClient`, `IpcServer` (Unix domain socket, JSON-lines) |
| `hub_common/message_types.py` | All `msg_type` constants and payload dataclasses (matches `ipc-protocol.md`) |
| `hub_common/config.py` | `HubConfig` loader from `/etc/exampen/hub.conf`, env-vars, or defaults |

## Ownership Declaration

- **Writes:** Nothing. This is a library with no durable state.
- **Reads from:** `/etc/exampen/hub.conf` (config file, read-only).
- **Never writes to:** Any file, database, or socket (the protocol classes are tools; callers own the connections).
- **Transactional boundaries:** None. Pure definitions and transport utilities.

## Usage

```python
from hub_common import IpcEnvelope, IpcClient, IpcServer, load_hub_config
from hub_common.message_types import STORE_WRITE_REQUEST, StoreWriteRequestPayload
```
