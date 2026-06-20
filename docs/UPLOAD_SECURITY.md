# Upload Security

All user-controlled uploads must declare a policy in
`backend/core/upload_security/policies.py` and a route mapping in
`backend/core/upload_security/routes.py`.

## Policy Control

Developers control default limits in `DEFAULT_UPLOAD_POLICIES`. Runtime
overrides use:

```text
UPLOAD_POLICY_<POLICY_ID>_<FIELD>
UPLOAD_POLICY_<POLICY_ID>_MAX_SIZE_MB
```

Examples:

```text
UPLOAD_POLICY_PDF_DOCUMENT_MAX_SIZE_MB=50
UPLOAD_POLICY_SUPPORT_MESSAGE_ATTACHMENT_MAX_FILES=10
UPLOAD_POLICY_HUB_RAW_DATA_BATCH_MAX_FRAMES_PER_BATCH=100000
```

Generic upload-security settings live in `config_async.py`:

```text
UPLOAD_SECURITY_ENABLED=true
UPLOAD_AV_ENABLED=true
UPLOAD_SCAN_REQUIRED=true
UPLOAD_AV_FAIL_CLOSED=true
CLAMAV_SOCKET=/var/run/clamav/clamd.ctl
CLAMD_HOST=127.0.0.1
CLAMD_PORT=3310
UPLOAD_PRIVATE_LOCAL_DIR=/var/lib/stoody/uploads
UPLOAD_QUARANTINE_PREFIX=quarantine
UPLOAD_RELEASED_PREFIX=clean
UPLOAD_REJECTED_PREFIX=rejected
UPLOAD_MAX_REQUEST_BODY_MB=64
UPLOAD_ALLOW_PUBLIC_LOCAL_FALLBACK=false
UPLOAD_ENABLE_PUBLIC_STATIC_MOUNT=false
```

Production local private storage is expected at:

```text
/var/lib/stoody/uploads/quarantine
/var/lib/stoody/uploads/clean
/var/lib/stoody/uploads/rejected
```

Those directories must be writable by the backend service user and readable by
the ClamAV daemon group. They must not be served by nginx.

## Binary Upload Order

Binary uploads must use `secure_upload()` or `secure_upload_many()`:

1. Read bytes with the policy size limit.
2. Check extension, declared MIME, and cheap magic bytes.
3. Write raw bytes to private quarantine storage.
4. Run malware scanning.
5. Run parser guards only after a clean scan.
6. Release clean bytes to private storage.
7. Persist an `upload_security_verdicts` record.

Production scanner errors fail closed when `UPLOAD_AV_FAIL_CLOSED=true`.
Scanner-disabled clean verdicts are only allowed when
`config_async.settings.DEBUG_MODE` is true.

Production ClamAV should run `clamav-daemon` and `clamav-freshclam`; verify
with `clamdscan --fdpass /etc/hosts`. Keep ClamAV `MaxScanSize`,
`MaxFileSize`, and `StreamMaxLength` aligned with the largest enabled binary
upload policy, or lower the affected `UPLOAD_POLICY_*_MAX_SIZE_MB` overrides.

nginx should keep `client_max_body_size 64m` unless a larger backend upload
policy is explicitly approved and the malware scanner limits are raised with it.

## Structured Uploads

Hub raw data and stroke uploads are JSON, not files. They use the same policy
registry for request-size, count, decoded-byte, and schema limits, but do not
run malware scanning.

## Static Coverage

`tests/test_upload_policy_coverage.py` scans FastAPI route handlers for
`UploadFile = File(...)`. A new upload route must be added to
`UPLOAD_ROUTE_POLICY_MAP` or the test fails. Exemptions are intentionally empty
and must include an owner and expiry if ever added.

## Public Storage

Production must not expose raw `backend/uploads` by default. The `/uploads`
static mount is controlled by `UPLOAD_ENABLE_PUBLIC_STATIC_MOUNT` and should
stay disabled in production. Raw user uploads go to private storage and should
be served through business endpoints or `/api/v1/uploads/{upload_id}/download`
with purpose-aware authorization.

The generic download endpoint only serves `status=clean` verdicts whose
`purpose_metadata.purpose` has an explicit authorizer. Tenant ownership alone is
not sufficient.
