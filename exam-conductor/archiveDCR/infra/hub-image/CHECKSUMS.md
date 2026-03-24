# ExamPen Hub Golden Image — Checksums

Verification checksums for released golden images. Always verify the
checksum after downloading and before flashing to ensure image integrity.

## Verification Commands

```bash
# Verify SHA-256 (preferred)
sha256sum -c exampen-hub-<version>.img.xz.sha256

# Verify MD5 (fallback)
md5sum -c exampen-hub-<version>.img.xz.md5
```

## Released Images

### v1.0.0

| File | Size | SHA-256 | MD5 |
|------|------|---------|-----|
| `exampen-hub-1.0.0.img.xz` | — | `<pending build>` | `<pending build>` |

**Build date:** —
**Base OS:** Ubuntu Server 24.04 LTS arm64
**BlueZ version:** —
**Python version:** 3.12.x

---

*Template: After each build, update this file with the actual checksums
from the `output/` directory. The build script generates `.sha256` and
`.md5` files automatically.*

### Template for New Releases

```
### v<VERSION>

| File | Size | SHA-256 | MD5 |
|------|------|---------|-----|
| `exampen-hub-<VERSION>.img.xz` | <SIZE> | `<SHA256>` | `<MD5>` |

**Build date:** <DATE>
**Base OS:** Ubuntu Server 24.04 LTS arm64
**BlueZ version:** <BLUEZ_VER>
**Python version:** 3.12.x
```
