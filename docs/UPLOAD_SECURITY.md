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
UPLOAD_REJECTED_RETENTION_DAYS=30
UPLOAD_QUARANTINE_RETENTION_HOURS=24
UPLOAD_SCANNER_TIMEOUT_SECONDS=30
UPLOAD_FRESHCLAM_MAX_AGE_HOURS=48
UPLOAD_MAX_REQUEST_BODY_MB=64
UPLOAD_DEPLOY_VALIDATION_STATUS_FILE=/home/ubuntu/backend/data/upload_deploy_validation_status.json
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

## Retention And Cleanup

Clean releases delete their quarantine copy after the file is copied to private
`clean/` or `derived/` storage. Rejected uploads remain under `rejected/` with a
metadata sidecar containing verdict and rejection reason.

Run cleanup in dry-run mode first:

```powershell
python scripts/cleanup_upload_storage.py
```

Then execute deletion of expired quarantine/rejected files:

```powershell
python scripts/cleanup_upload_storage.py --execute
```

Production deploys install a systemd timer for this cleanup:

```bash
systemctl is-enabled stoody-upload-cleanup.timer
systemctl is-active stoody-upload-cleanup.timer
systemctl list-timers stoody-upload-cleanup.timer
```

The timer defaults to daily execution around `03:20` with randomized delay and
runs:

```bash
python scripts/cleanup_upload_storage.py --execute
```

Override the timer from deployment with:

```bash
STOODY_UPLOAD_CLEANUP_ENABLED=true
STOODY_UPLOAD_CLEANUP_UNIT=stoody-upload-cleanup
STOODY_UPLOAD_CLEANUP_TIMER="*-*-* 03:20:00"
STOODY_UPLOAD_CLEANUP_RANDOMIZED_DELAY_SEC=15m
```

Useful options:

```text
--root /var/lib/stoody/uploads
--rejected-retention-days 30
--quarantine-retention-hours 24
```

The cleanup command only considers `quarantine/` and `rejected/`. It does not
delete `clean/` or `derived/` objects.

## Monitoring And Alerts

Prometheus metrics include:

```text
skillbot_upload_security_total{policy_id,outcome}
skillbot_upload_security_rejections_total{policy_id,reason}
skillbot_upload_security_scan_duration_seconds{policy_id,status}
skillbot_upload_security_alert_active{alert_type}
skillbot_upload_storage_bytes{prefix}
skillbot_upload_freshclam_age_seconds
skillbot_dependency_health{dependency="upload_malware_scanner"}
```

Expected alert types:

```text
scanner_unavailable
scanner_timeout
freshclam_stale
```

Alert on scanner unavailable, scanner timeout spikes, stale freshclam age,
infected/rejected spikes, and private upload storage growth. `/health` updates
scanner availability, freshclam age, and private storage usage gauges.

## Deployment Validation Gate

Run the deployment validation script after production deploys:

```powershell
python scripts/validate_upload_security_deploy.py
```

Set `BACKEND_HEALTH_URL` to the deployed backend health URL before running it.
The prod GitHub Actions deploy workflow runs this script after deployment and
fails the deploy if upload security is disabled or unhealthy. It checks:

```text
UPLOAD_SCAN_REQUIRED=true
UPLOAD_AV_FAIL_CLOSED=true
CLAMAV_SOCKET exists
CLAMAV socket is clamav:clamav, mode 660, and not world-accessible
backend service user is in the clamav group
clamav-daemon active
clamav-freshclam active
EICAR is detected
private quarantine/rejected/clean directories exist
stoody-upload-cleanup.timer is enabled and active
nginx config does not alias private upload root
backend health reports upload_malware_scanner.available=true
```

## ClamAV Socket Hardening

Target production socket posture:

```text
LocalSocket /var/run/clamav/clamd.ctl
LocalSocketMode 660
LocalSocketGroup clamav
```

The backend service user should be in the scanner-access group:

```bash
sudo usermod -aG clamav ubuntu
sudo mkdir -p /etc/systemd/system/clamav-daemon.socket.d
printf '[Socket]\nSocketMode=0660\nSocketUser=clamav\nSocketGroup=clamav\n' \
  | sudo tee /etc/systemd/system/clamav-daemon.socket.d/upload-security.conf
sudo systemctl daemon-reload
sudo systemctl stop clamav-daemon.service clamav-daemon.socket
sudo systemctl start clamav-daemon.socket
sudo systemctl restart clamav-daemon
sudo systemctl restart stoody-backend
```

The systemd socket drop-in is required on hosts where
`clamav-daemon.socket` owns `/run/clamav/clamd.ctl`; otherwise `clamd.conf`
can say `LocalSocketMode 660` while systemd still creates a `666` socket.

Verify:

```bash
stat -c '%U:%G %a %n' /var/run/clamav/clamd.ctl
id ubuntu
clamdscan --fdpass /etc/hosts
```

Non-service local users should not have access to the ClamAV socket.

## Grafana Monitoring

The production monitoring dashboard must show both upload activity and
effective upload-security configuration.

Required dashboard groups:

- Upload policy catalog from `skillbot_upload_policy_limit` and
  `skillbot_upload_policy_info`.
- Route-to-policy coverage from `skillbot_upload_route_policy_info`.
- Runtime security toggles from `skillbot_upload_runtime_config`.
- Scanner availability from
  `skillbot_dependency_health{dependency="upload_malware_scanner"}`.
- Freshclam age from `skillbot_upload_freshclam_age_seconds`.
- Rejection and abuse trends from
  `skillbot_upload_security_rejections_total`.
- Quarantine, rejected, clean, and derived storage from
  `skillbot_upload_storage_bytes`.
- Last production deploy validation result from
  `skillbot_upload_deploy_validation` and
  `skillbot_upload_deploy_validation_check`.

Prometheus and exporter ports must not be public. Grafana must be authenticated
and served over HTTPS. Backend metrics access should be restricted to the
monitoring host or another explicitly authorized scraper. In production,
`METRICS_ACCESS_TOKEN` should be set and Prometheus should send it as a bearer
token from a server-only secret file.

## Operations Runbook

Check ClamAV status:

```bash
systemctl is-active clamav-daemon
systemctl is-active clamav-freshclam
journalctl -u clamav-daemon -n 100 --no-pager
```

Update signatures:

```bash
sudo freshclam
systemctl status clamav-freshclam --no-pager
```

Inspect rejected verdicts in Mongo:

```javascript
db.upload_security_verdicts.find({status: {$in: ["rejected", "scan_failed"]}})
  .sort({created_at: -1})
  .limit(20)
```

If uploads fail closed, check `/health`, ClamAV socket permissions, daemon
status, freshclam freshness, and `skillbot_upload_security_alert_active`.

Rotate or repair private upload directory permissions:

```bash
sudo chown -R ubuntu:clamav /var/lib/stoody/uploads
sudo find /var/lib/stoody/uploads -type d -exec chmod 2750 {} \;
sudo find /var/lib/stoody/uploads -type f -exec chmod 0640 {} \;
```

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
