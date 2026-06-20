# Jitsi Grafana Prometheus Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy a secured lightweight Grafana/Prometheus/Alertmanager monitoring stack on the Jitsi EC2 and extend backend metrics so Grafana shows both upload-security activity and effective runtime configuration.

**Architecture:** Run the monitoring stack on the existing Jitsi EC2 as a controlled MVP, with Prometheus retention/resource limits because the host has 2 vCPU, 3.7 GiB RAM, and a 19 GiB root disk. Grafana is the only UI exposed through Caddy over HTTPS with external auth plus Grafana login; Prometheus and Alertmanager stay bound to localhost/private Docker networking. Backend runtime config and upload policy state are exported as non-secret Prometheus metrics from the backend, while host/service status comes from node exporter plus systemd/textfile collectors.

**Tech Stack:** Docker Compose, Prometheus, Grafana OSS, Alertmanager, blackbox exporter, node exporter, Caddy, FastAPI, prometheus-client, GitHub Actions backend deploy, Ubuntu 24.04 on `ubuntu@3.6.147.238`.

---

## Current Facts

- Jitsi host: `ubuntu@3.6.147.238`
- Jitsi domain: `class.stoody.in`
- Working local SSH key: `D:\SOFTWARE_Projects_LP\skiller-bot\stoody-board.pem`
- Jitsi server capacity checked on 2026-06-20:
  - OS: Ubuntu 24.04.3 LTS
  - CPU: 2 vCPU
  - RAM: 3.7 GiB
  - Disk: 19 GiB root, 8.0 GiB used, 11 GiB free
  - Jitsi runs as Docker containers behind Caddy
  - Prometheus and Grafana are not currently running there
- Existing local monitoring scaffold:
  - `grafana-main-app/docker-compose.yml`
  - `grafana-main-app/prometheus/prometheus.yml`
  - `grafana-main-app/prometheus/rules/main-app-alerts.yml`
  - `grafana-main-app/grafana/dashboards/*.json`
- Existing backend upload-security metrics:
  - `skillbot_upload_security_total{policy_id,outcome}`
  - `skillbot_upload_security_rejections_total{policy_id,reason}`
  - `skillbot_upload_security_scan_duration_seconds{policy_id,status}`
  - `skillbot_upload_security_alert_active{alert_type}`
  - `skillbot_upload_storage_bytes{prefix}`
  - `skillbot_upload_freshclam_age_seconds`
  - `skillbot_dependency_health{dependency="upload_malware_scanner"}`
- Existing upload policy source of truth:
  - `backend/core/upload_security/policies.py`
  - `backend/core/upload_security/routes.py`
- Existing backend metrics endpoints:
  - `/metrics`
  - `/api/metrics`
  - `/api/v1/metrics`

## Security Decisions

- Do not expose Prometheus, Alertmanager, node exporter, or backend host exporters publicly.
- Bind Prometheus and Alertmanager to `127.0.0.1` on the Jitsi host or leave them Docker-internal only.
- Expose only Grafana through Caddy.
- Put Grafana behind two layers:
  - Caddy HTTPS + `basic_auth`
  - Grafana login with a strong admin password from an env file
- Disable anonymous Grafana access.
- Do not export secret values into metrics. Export booleans, effective numeric limits, labels for policy IDs/routes, and safe status strings only.
- Restrict backend `/api/metrics` access so only the monitoring host can scrape it, either by backend Nginx allowlist/security group or a metrics token if code support is added.
- Keep monitoring data retention small on the Jitsi host until the instance/disk is upgraded.

## File Structure

### Backend Repo Files

- Modify: `backend/core/observability.py`
  - Add gauges/info metrics for effective upload policy config, route policy map, runtime security config, and upload deploy validation status.
- Create: `backend/core/upload_security/metrics_exporter.py`
  - Collect safe upload policy and route-map state from `policies.py`, `routes.py`, and `config_async.py`.
- Modify: `backend/main_async.py`
  - Register/update config metrics during startup and `/health`, without leaking secrets.
- Create: `backend/tests/test_upload_security_metrics_exporter.py`
  - Verify exported policy limits, route mappings, env override indicators, and no secret values.
- Modify: `backend/docs/UPLOAD_SECURITY.md`
  - Document Grafana metrics, dashboard panels, and safe/non-secret config export.

### Monitoring Stack Files

- Modify: `grafana-main-app/docker-compose.yml`
  - Make stack suitable for Jitsi host MVP: bind sensitive ports to localhost, reduce Prometheus retention, add node exporter, remove or disable Loki/Promtail unless explicitly needed.
- Modify: `grafana-main-app/prometheus/prometheus.yml`
  - Add backend scrape, Jitsi node exporter scrape, blackbox probes, and localhost-only targets.
- Modify: `grafana-main-app/prometheus/rules/main-app-alerts.yml`
  - Add alerts for backend upload config drift, scanner unavailable, Jitsi disk/RAM pressure, Prometheus target down, and Grafana auth exposure.
- Create: `grafana-main-app/grafana/dashboards/upload-security-posture.json`
  - Dashboard for upload policy catalog, runtime config, scanner status, rejection trends, and storage.
- Create: `grafana-main-app/grafana/dashboards/jitsi-monitoring-host.json`
  - Dashboard for Jitsi host resources, Jitsi containers, Caddy, Docker, and monitoring stack health.
- Create: `grafana-main-app/.env.example`
  - Safe template for Grafana admin user/password, backend target, retention, and optional alert receiver settings.
- Create: `grafana-main-app/deploy/jitsi-caddy-monitoring.example.Caddyfile`
  - Caddy route example for `monitor.stoody.in` with HTTPS and `basic_auth`.
- Create: `grafana-main-app/docs/JITSI_MONITORING_DEPLOY.md`
  - Direct server steps, rollback, validation commands, DNS requirements, and security checks.

### Direct Jitsi Server Files

- Create: `/opt/stoody-monitoring`
  - Deployment directory for copied `grafana-main-app` stack.
- Create: `/opt/stoody-monitoring/.env`
  - Server-only secrets and runtime config. Must not be committed.
- Modify: `/etc/caddy/Caddyfile`
  - Add authenticated reverse proxy for Grafana only.
- Install package/service:
  - `prometheus-node-exporter` or run `prom/node-exporter` as a Docker container.

---

## Task 1: Make The Monitoring Stack Deployable For The Jitsi Host

**Files:**
- Modify: `grafana-main-app/docker-compose.yml`
- Create: `grafana-main-app/.env.example`
- Test: `docker compose -f grafana-main-app/docker-compose.yml config`

- [ ] **Step 1: Back up the current local compose behavior**

Run:

```powershell
git -C backend status --short
docker compose -f grafana-main-app/docker-compose.yml config
```

Expected:

```text
docker compose renders a valid config
```

- [ ] **Step 2: Adjust `grafana-main-app/docker-compose.yml` for Jitsi-host MVP**

Change the stack to:

```yaml
services:
  prometheus:
    image: prom/prometheus:v2.54.1
    container_name: stoody-monitoring-prometheus
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.path=/prometheus
      - --storage.tsdb.retention.time=${PROMETHEUS_RETENTION_TIME:-7d}
      - --storage.tsdb.retention.size=${PROMETHEUS_RETENTION_SIZE:-4GB}
      - --web.enable-lifecycle
    ports:
      - "127.0.0.1:9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus/rules:/etc/prometheus/rules:ro
      - ./data/prometheus:/prometheus
    restart: unless-stopped
    depends_on:
      - blackbox-exporter
      - node-exporter
      - alertmanager

  grafana:
    image: grafana/grafana-oss:11.2.2
    container_name: stoody-monitoring-grafana
    ports:
      - "127.0.0.1:3000:3000"
    environment:
      GF_SECURITY_ADMIN_USER: ${GRAFANA_ADMIN_USER:?set GRAFANA_ADMIN_USER}
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:?set GRAFANA_ADMIN_PASSWORD}
      GF_SERVER_ROOT_URL: ${GRAFANA_ROOT_URL:?set GRAFANA_ROOT_URL}
      GF_USERS_DEFAULT_THEME: light
      GF_AUTH_ANONYMOUS_ENABLED: "false"
      GF_SECURITY_DISABLE_GRAVATAR: "true"
      GF_SECURITY_COOKIE_SECURE: "true"
      GF_SECURITY_COOKIE_SAMESITE: strict
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
      - ./data/grafana:/var/lib/grafana
    restart: unless-stopped
    depends_on:
      - prometheus

  alertmanager:
    image: prom/alertmanager:v0.27.0
    container_name: stoody-monitoring-alertmanager
    command:
      - --config.file=/etc/alertmanager/alertmanager.yml
      - --storage.path=/alertmanager
    ports:
      - "127.0.0.1:9093:9093"
    volumes:
      - ./alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - ./data/alertmanager:/alertmanager
    restart: unless-stopped

  blackbox-exporter:
    image: prom/blackbox-exporter:v0.25.0
    container_name: stoody-monitoring-blackbox
    command:
      - --config.file=/etc/blackbox/blackbox.yml
    ports:
      - "127.0.0.1:9115:9115"
    volumes:
      - ./blackbox/blackbox.yml:/etc/blackbox/blackbox.yml:ro
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:v1.8.2
    container_name: stoody-monitoring-node-exporter
    command:
      - --path.rootfs=/host
      - --collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)
    ports:
      - "127.0.0.1:9100:9100"
    volumes:
      - /:/host:ro,rslave
    pid: host
    restart: unless-stopped
```

- [ ] **Step 3: Create `grafana-main-app/.env.example`**

Use:

```env
GRAFANA_ADMIN_USER=stoody-admin
GRAFANA_ADMIN_PASSWORD=change-this-before-deploy
GRAFANA_ROOT_URL=https://monitor.stoody.in
PROMETHEUS_RETENTION_TIME=7d
PROMETHEUS_RETENTION_SIZE=4GB
BACKEND_METRICS_TARGET=api.stoody.in
BACKEND_HEALTH_URL=https://api.stoody.in/api/health
FRONTEND_HEALTH_URL=https://app.stoody.in/
JITSI_HEALTH_URL=https://class.stoody.in/
```

- [ ] **Step 4: Validate compose locally**

Run:

```powershell
docker compose -f grafana-main-app/docker-compose.yml --env-file grafana-main-app/.env.example config
```

Expected:

```text
No parse errors. Prometheus, Grafana, Alertmanager, blackbox-exporter, and node-exporter are present.
```

---

## Task 2: Add Safe Backend Runtime Config Metrics

**Files:**
- Modify: `backend/core/observability.py`
- Create: `backend/core/upload_security/metrics_exporter.py`
- Modify: `backend/main_async.py`
- Test: `backend/tests/test_upload_security_metrics_exporter.py`

- [ ] **Step 1: Write tests for safe upload policy/config export**

Create `backend/tests/test_upload_security_metrics_exporter.py`:

```python
from core.upload_security.metrics_exporter import build_upload_security_metric_rows


def test_upload_security_metric_rows_include_policy_limits():
    rows = build_upload_security_metric_rows()

    policy_rows = [row for row in rows if row["metric"] == "policy_limit"]
    assert any(
        row["labels"]["policy_id"] == "pdf_document"
        and row["labels"]["field"] == "max_size_bytes"
        and row["value"] > 0
        for row in policy_rows
    )


def test_upload_security_metric_rows_include_route_mappings():
    rows = build_upload_security_metric_rows()

    route_rows = [row for row in rows if row["metric"] == "route_policy"]
    assert any(
        row["labels"]["policy_id"] == "hub_raw_data_batch"
        and row["labels"]["method"] == "POST"
        for row in route_rows
    )


def test_upload_security_metric_rows_do_not_export_secrets():
    rows = build_upload_security_metric_rows()
    rendered = repr(rows).lower()

    forbidden_fragments = [
        "secret",
        "password",
        "token",
        "jwt",
        "mongodb_uri",
        "api_key",
        "openai",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in rendered
```

- [ ] **Step 2: Run test and verify it fails before implementation**

Run:

```powershell
cd backend
venv\Scripts\python -m pytest tests/test_upload_security_metrics_exporter.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'core.upload_security.metrics_exporter'
```

- [ ] **Step 3: Add metric definitions to `backend/core/observability.py`**

Add:

```python
UPLOAD_POLICY_LIMIT = Gauge(
    "skillbot_upload_policy_limit",
    "Effective upload policy numeric limits where labels identify the policy and field.",
    ["policy_id", "field"],
)

UPLOAD_POLICY_INFO = Gauge(
    "skillbot_upload_policy_info",
    "Effective upload policy metadata; value is always 1.",
    ["policy_id", "policy_kind", "allowed_extensions", "allowed_mime_types", "allowed_magic_types"],
)

UPLOAD_ROUTE_POLICY_INFO = Gauge(
    "skillbot_upload_route_policy_info",
    "Upload route to policy mapping; value is always 1.",
    ["method", "path_template", "policy_id", "owner_note"],
)

UPLOAD_RUNTIME_CONFIG = Gauge(
    "skillbot_upload_runtime_config",
    "Safe effective upload runtime config values where field identifies the setting.",
    ["field"],
)
```

Add helper:

```python
def set_upload_security_config_metric(metric: str, labels: dict[str, str], value: float) -> None:
    safe_labels = {key: _safe(val) for key, val in labels.items()}
    if metric == "policy_limit":
        UPLOAD_POLICY_LIMIT.labels(**safe_labels).set(value)
    elif metric == "policy_info":
        UPLOAD_POLICY_INFO.labels(**safe_labels).set(value)
    elif metric == "route_policy":
        UPLOAD_ROUTE_POLICY_INFO.labels(**safe_labels).set(value)
    elif metric == "runtime_config":
        UPLOAD_RUNTIME_CONFIG.labels(**safe_labels).set(value)
```

- [ ] **Step 4: Create `backend/core/upload_security/metrics_exporter.py`**

Use:

```python
"""Safe Prometheus export for upload-security runtime configuration."""

from __future__ import annotations

from typing import Any

import config_async as settings

from .policies import DEFAULT_UPLOAD_POLICIES, get_upload_policy
from .routes import UPLOAD_ROUTE_POLICY_MAP

_NUMERIC_POLICY_FIELDS = (
    "max_size_bytes",
    "max_total_size_bytes",
    "max_files",
    "max_pdf_pages",
    "max_image_pixels",
    "max_archive_entries",
    "max_archive_depth",
    "max_archive_uncompressed_bytes",
    "max_rows",
    "max_columns",
    "max_json_bytes",
    "max_json_depth",
    "max_json_fields",
    "max_frames_per_batch",
    "max_points_per_frame",
    "max_strokes_per_request",
)


def build_upload_security_metric_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for policy_id in sorted(DEFAULT_UPLOAD_POLICIES):
        policy = get_upload_policy(policy_id)
        rows.append(
            {
                "metric": "policy_info",
                "labels": {
                    "policy_id": policy.policy_id,
                    "policy_kind": policy.policy_kind,
                    "allowed_extensions": ",".join(policy.allowed_extensions or ()),
                    "allowed_mime_types": ",".join(policy.allowed_mime_types or ()),
                    "allowed_magic_types": ",".join(policy.allowed_magic_types or ()),
                },
                "value": 1.0,
            }
        )
        for field_name in _NUMERIC_POLICY_FIELDS:
            value = getattr(policy, field_name, None)
            if value is not None:
                rows.append(
                    {
                        "metric": "policy_limit",
                        "labels": {"policy_id": policy.policy_id, "field": field_name},
                        "value": float(value),
                    }
                )

    for route in UPLOAD_ROUTE_POLICY_MAP:
        rows.append(
            {
                "metric": "route_policy",
                "labels": {
                    "method": route.method,
                    "path_template": route.path_template,
                    "policy_id": route.policy_id,
                    "owner_note": route.owner_note,
                },
                "value": 1.0,
            }
        )

    runtime_config = {
        "upload_security_enabled": bool(settings.UPLOAD_SECURITY_ENABLED),
        "upload_av_enabled": bool(settings.UPLOAD_AV_ENABLED),
        "upload_scan_required": bool(settings.UPLOAD_SCAN_REQUIRED),
        "upload_av_fail_closed": bool(settings.UPLOAD_AV_FAIL_CLOSED),
        "upload_enable_public_static_mount": bool(settings.UPLOAD_ENABLE_PUBLIC_STATIC_MOUNT),
        "upload_allow_public_local_fallback": bool(settings.UPLOAD_ALLOW_PUBLIC_LOCAL_FALLBACK),
        "upload_max_request_body_mb": float(settings.UPLOAD_MAX_REQUEST_BODY_MB),
        "upload_scanner_timeout_seconds": float(settings.UPLOAD_SCANNER_TIMEOUT_SECONDS),
        "upload_freshclam_max_age_hours": float(settings.UPLOAD_FRESHCLAM_MAX_AGE_HOURS),
    }
    for field, value in runtime_config.items():
        rows.append(
            {
                "metric": "runtime_config",
                "labels": {"field": field},
                "value": float(value),
            }
        )

    return rows
```

- [ ] **Step 5: Wire exporter in `backend/main_async.py`**

Import:

```python
from core.observability import set_upload_security_config_metric
from core.upload_security.metrics_exporter import build_upload_security_metric_rows
```

Add helper:

```python
def _refresh_upload_security_config_metrics() -> None:
    try:
        for row in build_upload_security_metric_rows():
            set_upload_security_config_metric(
                metric=row["metric"],
                labels=row["labels"],
                value=row["value"],
            )
    except Exception as exc:
        logger.warning("Upload security config metrics refresh failed: %s", exc)
```

Call `_refresh_upload_security_config_metrics()`:

```python
_refresh_upload_security_config_metrics()
```

Place the call during app startup after settings are loaded and inside `/health` before returning, so Prometheus sees fresh effective config after env changes/restarts.

- [ ] **Step 6: Run focused tests**

Run:

```powershell
cd backend
venv\Scripts\python -m pytest tests/test_upload_security_metrics_exporter.py tests/test_upload_policy_coverage.py -q
```

Expected:

```text
All selected tests pass.
```

- [ ] **Step 7: Confirm metrics render locally or in deployed backend after CI/CD**

Run after deploy:

```bash
curl -fsS https://api.stoody.in/api/metrics | grep -E 'skillbot_upload_policy_limit|skillbot_upload_route_policy_info|skillbot_upload_runtime_config' | head -40
```

Expected:

```text
Metrics show policy limits, route mappings, and runtime config booleans/numbers.
No secret values appear.
```

---

## Task 3: Configure Prometheus Scrapes And Alerts

**Files:**
- Modify: `grafana-main-app/prometheus/prometheus.yml`
- Modify: `grafana-main-app/prometheus/rules/main-app-alerts.yml`
- Test: `docker run --rm -v "$PWD/grafana-main-app/prometheus:/etc/prometheus" prom/prometheus:v2.54.1 promtool check config /etc/prometheus/prometheus.yml`

- [ ] **Step 1: Update `prometheus.yml` for Jitsi-host deployment**

Use:

```yaml
global:
  scrape_interval: 30s
  evaluation_interval: 30s

rule_files:
  - /etc/prometheus/rules/*.yml

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

scrape_configs:
  - job_name: prometheus
    static_configs:
      - targets: ["prometheus:9090"]

  - job_name: jitsi_monitoring_host_node
    static_configs:
      - targets: ["node-exporter:9100"]
        labels:
          server: jitsi-monitoring
          public_ip: "3.6.147.238"

  - job_name: skillbot_backend_metrics
    scheme: https
    metrics_path: /api/metrics
    static_configs:
      - targets: ["${BACKEND_METRICS_TARGET:-api.stoody.in}"]

  - job_name: skillbot_backend_health
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets: ["${BACKEND_HEALTH_URL:-https://api.stoody.in/api/health}"]
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox-exporter:9115

  - job_name: stoody_frontend_health
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets: ["${FRONTEND_HEALTH_URL:-https://app.stoody.in/}"]
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox-exporter:9115

  - job_name: jitsi_public_health
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets: ["${JITSI_HEALTH_URL:-https://class.stoody.in/}"]
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox-exporter:9115

  - job_name: blackbox_exporter
    static_configs:
      - targets: ["blackbox-exporter:9115"]
```

Note: Prometheus does not expand shell variables inside YAML by default. During implementation either generate this file from a template or keep literal production values in the deployed file:

```yaml
targets: ["api.stoody.in"]
targets: ["https://api.stoody.in/api/health"]
targets: ["https://app.stoody.in/"]
targets: ["https://class.stoody.in/"]
```

- [ ] **Step 2: Add upload config drift alerts**

Append to `main-app-alerts.yml`:

```yaml
      - alert: SkillbotUploadScanNotRequired
        expr: skillbot_upload_runtime_config{field="upload_scan_required"} != 1
        for: 2m
        labels:
          severity: critical
          service: skillbot-backend
          area: upload-security
        annotations:
          summary: "Upload scan requirement is disabled"
          description: "UPLOAD_SCAN_REQUIRED is not effectively true in backend runtime metrics."

      - alert: SkillbotUploadFailClosedDisabled
        expr: skillbot_upload_runtime_config{field="upload_av_fail_closed"} != 1
        for: 2m
        labels:
          severity: critical
          service: skillbot-backend
          area: upload-security
        annotations:
          summary: "Upload fail-closed is disabled"
          description: "UPLOAD_AV_FAIL_CLOSED is not effectively true in backend runtime metrics."

      - alert: SkillbotUploadPublicStaticMountEnabled
        expr: skillbot_upload_runtime_config{field="upload_enable_public_static_mount"} != 0
        for: 2m
        labels:
          severity: critical
          service: skillbot-backend
          area: upload-security
        annotations:
          summary: "Public upload static mount is enabled"
          description: "UPLOAD_ENABLE_PUBLIC_STATIC_MOUNT should be false in production."

      - alert: JitsiMonitoringHostDiskPressure
        expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) < 0.20
        for: 10m
        labels:
          severity: warning
          service: jitsi-monitoring-host
        annotations:
          summary: "Jitsi monitoring host root disk below 20%"
          description: "The monitoring MVP shares the Jitsi host; increase disk or reduce retention."

      - alert: JitsiMonitoringHostMemoryPressure
        expr: (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) < 0.15
        for: 10m
        labels:
          severity: warning
          service: jitsi-monitoring-host
        annotations:
          summary: "Jitsi monitoring host memory pressure"
          description: "Available memory is below 15%; upgrade instance or move monitoring."
```

- [ ] **Step 3: Validate Prometheus config**

Run:

```powershell
docker run --rm -v "${PWD}/grafana-main-app/prometheus:/etc/prometheus" prom/prometheus:v2.54.1 promtool check config /etc/prometheus/prometheus.yml
```

Expected:

```text
SUCCESS: /etc/prometheus/prometheus.yml is valid prometheus config file syntax
```

---

## Task 4: Add Dashboards For Security Decisions And Host Status

**Files:**
- Create: `grafana-main-app/grafana/dashboards/upload-security-posture.json`
- Create: `grafana-main-app/grafana/dashboards/jitsi-monitoring-host.json`
- Modify: `grafana-main-app/grafana/provisioning/dashboards/dashboards.yml`

- [ ] **Step 1: Add `Upload Security Posture` dashboard**

Dashboard panels must include:

```text
Policy Catalog Table:
- policy_id
- route method/path
- max_size_bytes
- max_total_size_bytes
- allowed extensions/MIME/magic labels

Runtime Config Stat Panels:
- upload_scan_required
- upload_av_fail_closed
- upload_security_enabled
- upload_enable_public_static_mount
- upload_allow_public_local_fallback

Scanner Health:
- skillbot_dependency_health{dependency="upload_malware_scanner"}
- skillbot_upload_freshclam_age_seconds
- skillbot_upload_security_alert_active

Upload Abuse:
- rate(skillbot_upload_security_rejections_total[5m]) by policy_id/reason
- increase(skillbot_upload_security_rejections_total{reason="infected"}[24h])
- histogram_quantile(0.95, skillbot_upload_security_scan_duration_seconds_bucket)

Storage:
- skillbot_upload_storage_bytes by prefix
```

- [ ] **Step 2: Add `Jitsi Monitoring Host` dashboard**

Dashboard panels must include:

```text
Host Resources:
- CPU load
- memory available
- root disk free
- filesystem usage

Monitoring Stack:
- up{job="prometheus"}
- up{job="jitsi_monitoring_host_node"}
- up{job="skillbot_backend_metrics"}
- up{job="skillbot_backend_health"}

Jitsi Availability:
- probe_success{job="jitsi_public_health"}
- probe_duration_seconds{job="jitsi_public_health"}

Capacity:
- node_filesystem_avail_bytes{mountpoint="/"}
- node_memory_MemAvailable_bytes
- node_load1
```

- [ ] **Step 3: Confirm provisioning picks up dashboards**

Run:

```powershell
docker compose -f grafana-main-app/docker-compose.yml --env-file grafana-main-app/.env.example up -d grafana prometheus
```

Expected:

```text
Grafana starts and dashboards appear under provisioned dashboards.
```

Stop local stack after validation:

```powershell
docker compose -f grafana-main-app/docker-compose.yml --env-file grafana-main-app/.env.example down
```

---

## Task 5: Secure Grafana Access With Caddy

**Files:**
- Create: `grafana-main-app/deploy/jitsi-caddy-monitoring.example.Caddyfile`
- Direct server modify: `/etc/caddy/Caddyfile`

- [ ] **Step 1: Decide DNS name**

Preferred:

```text
monitor.stoody.in A 3.6.147.238
```

Fallback if DNS is not ready:

```text
Do not expose Grafana publicly.
Use SSH tunnel:
ssh -i stoody-board.pem -L 3000:127.0.0.1:3000 ubuntu@3.6.147.238
Open http://127.0.0.1:3000 locally.
```

- [ ] **Step 2: Generate Caddy basic auth hash on the server**

Run on Jitsi server:

```bash
caddy hash-password --plaintext '<strong-random-password>'
```

Expected:

```text
$2a$14$...
```

- [ ] **Step 3: Create Caddy example file**

Create `grafana-main-app/deploy/jitsi-caddy-monitoring.example.Caddyfile`:

```caddyfile
monitor.stoody.in {
    encode gzip zstd

    basic_auth {
        stoody-admin <caddy-hashed-password>
    }

    reverse_proxy 127.0.0.1:3000

    header {
        X-Content-Type-Options nosniff
        X-Frame-Options DENY
        Referrer-Policy no-referrer
    }
}
```

- [ ] **Step 4: Apply Caddy config on the Jitsi server**

Run:

```bash
sudo cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak.$(date -u +%Y%m%dT%H%M%SZ)
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
sudo systemctl status caddy --no-pager
```

Expected:

```text
Config valid
caddy.service active (running)
```

- [ ] **Step 5: Verify ports are not exposed**

Run on Jitsi server:

```bash
ss -ltnp | grep -E ':3000|:9090|:9093|:9100|:9115'
```

Expected:

```text
Each service is bound to 127.0.0.1, not 0.0.0.0.
```

---

## Task 6: Deploy The Monitoring Stack On The Jitsi Server

**Files:**
- Direct server create: `/opt/stoody-monitoring`
- Direct server create: `/opt/stoody-monitoring/.env`

- [ ] **Step 1: Create deployment directory**

Run:

```bash
sudo mkdir -p /opt/stoody-monitoring
sudo chown -R ubuntu:ubuntu /opt/stoody-monitoring
chmod 750 /opt/stoody-monitoring
```

- [ ] **Step 2: Copy monitoring stack**

From local machine:

```powershell
scp -i stoody-board.pem -r grafana-main-app\* ubuntu@3.6.147.238:/opt/stoody-monitoring/
```

- [ ] **Step 3: Create server-only `.env`**

On Jitsi server:

```bash
cd /opt/stoody-monitoring
install -m 600 /dev/null .env
cat > .env <<'EOF'
GRAFANA_ADMIN_USER=stoody-admin
GRAFANA_ADMIN_PASSWORD=<strong-random-password>
GRAFANA_ROOT_URL=https://monitor.stoody.in
PROMETHEUS_RETENTION_TIME=7d
PROMETHEUS_RETENTION_SIZE=4GB
BACKEND_METRICS_TARGET=api.stoody.in
BACKEND_HEALTH_URL=https://api.stoody.in/api/health
FRONTEND_HEALTH_URL=https://app.stoody.in/
JITSI_HEALTH_URL=https://class.stoody.in/
EOF
chmod 600 .env
```

- [ ] **Step 4: Start stack**

Run:

```bash
cd /opt/stoody-monitoring
docker compose --env-file .env up -d
docker compose ps
```

Expected:

```text
prometheus, grafana, alertmanager, blackbox-exporter, node-exporter are Up
```

- [ ] **Step 5: Validate local services**

Run:

```bash
curl -fsS http://127.0.0.1:9090/-/ready
curl -fsS http://127.0.0.1:3000/api/health
curl -fsS http://127.0.0.1:9115/-/healthy
curl -fsS http://127.0.0.1:9100/metrics | head
```

Expected:

```text
Prometheus ready
Grafana database ok
Blackbox healthy
Node exporter metrics render
```

---

## Task 7: Secure Backend Metrics Scraping

**Files:**
- Direct backend server config if Nginx controls `/api/metrics`
- Optional backend code later if token-based metrics auth is preferred

- [ ] **Step 1: Check current backend metrics exposure**

Run from local:

```powershell
curl.exe -I https://api.stoody.in/api/metrics
```

Expected secure target:

```text
Not publicly readable unless requester is the monitoring host or authorized.
```

- [ ] **Step 2: Prefer network allowlist**

On the backend Nginx host, restrict `/api/metrics` to:

```nginx
location = /api/metrics {
    allow 3.6.147.238;
    deny all;
    proxy_pass http://127.0.0.1:<backend-port>/api/metrics;
}
```

Also apply equivalent blocks for:

```nginx
/metrics
/api/v1/metrics
```

- [ ] **Step 3: Validate from Jitsi server**

Run:

```bash
curl -fsS https://api.stoody.in/api/metrics | head
```

Expected:

```text
Prometheus metrics render from Jitsi server.
```

- [ ] **Step 4: Validate from non-allowed source**

Run from local workstation:

```powershell
curl.exe -I https://api.stoody.in/api/metrics
```

Expected:

```text
403 Forbidden
```

---

## Task 8: Validate Dashboard Data And Alerts

**Files:**
- No file changes unless dashboards/alerts need correction

- [ ] **Step 1: Check Prometheus targets**

Run on Jitsi server:

```bash
curl -fsS http://127.0.0.1:9090/api/v1/targets | python3 -m json.tool | grep -E '"job"|"health"|"lastError"'
```

Expected:

```text
skillbot_backend_metrics, skillbot_backend_health, jitsi_monitoring_host_node, blackbox_exporter are up.
```

- [ ] **Step 2: Query key upload security metrics**

Run:

```bash
curl -G -fsS http://127.0.0.1:9090/api/v1/query --data-urlencode 'query=skillbot_upload_runtime_config'
curl -G -fsS http://127.0.0.1:9090/api/v1/query --data-urlencode 'query=skillbot_upload_policy_limit'
curl -G -fsS http://127.0.0.1:9090/api/v1/query --data-urlencode 'query=skillbot_upload_route_policy_info'
```

Expected:

```text
Each query returns success and at least one time series.
```

- [ ] **Step 3: Verify Grafana authentication**

Run from local:

```powershell
curl.exe -I https://monitor.stoody.in
```

Expected:

```text
401 Unauthorized from Caddy basic_auth before Grafana login.
```

Then login through browser using:

```text
Caddy basic auth credentials
Grafana admin credentials
```

- [ ] **Step 4: Verify no sensitive ports are public**

Run from local:

```powershell
Test-NetConnection 3.6.147.238 -Port 9090
Test-NetConnection 3.6.147.238 -Port 9093
Test-NetConnection 3.6.147.238 -Port 9100
Test-NetConnection 3.6.147.238 -Port 9115
Test-NetConnection 3.6.147.238 -Port 3000
```

Expected:

```text
TcpTestSucceeded: False
```

- [ ] **Step 5: Check host pressure after deployment**

Run on Jitsi server:

```bash
free -h
df -h /
docker stats --no-stream
uptime
```

Expected:

```text
At least 1 GiB memory available after cache.
Root disk remains below 70% used.
Load remains low when no active classes are running.
```

---

## Task 9: Document Operations And Rollback

**Files:**
- Create: `grafana-main-app/docs/JITSI_MONITORING_DEPLOY.md`
- Modify: `backend/docs/UPLOAD_SECURITY.md`

- [ ] **Step 1: Create deploy runbook**

Include:

```markdown
# Jitsi Monitoring Deployment Runbook

## Host

- SSH: ubuntu@3.6.147.238
- Deployment dir: /opt/stoody-monitoring
- Public UI: https://monitor.stoody.in

## Start

cd /opt/stoody-monitoring
docker compose --env-file .env up -d

## Stop

cd /opt/stoody-monitoring
docker compose down

## Validate

curl -fsS http://127.0.0.1:9090/-/ready
curl -fsS http://127.0.0.1:3000/api/health
docker compose ps
df -h /
free -h

## Rollback

cd /opt/stoody-monitoring
docker compose down
sudo cp /etc/caddy/Caddyfile.bak.<timestamp> /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy

## Security

- Prometheus, Alertmanager, node-exporter, blackbox-exporter bind to localhost only.
- Grafana is behind Caddy basic_auth and Grafana login.
- Metrics endpoints on backend are allowlisted to the monitoring host.
- No secret values are exported as Prometheus labels or metric values.
```

- [ ] **Step 2: Update upload security docs**

Add a section to `backend/docs/UPLOAD_SECURITY.md`:

```markdown
## Grafana Monitoring

The production monitoring dashboard must show both upload activity and effective upload-security configuration.

Required dashboard groups:

- Upload policy catalog from `skillbot_upload_policy_limit` and `skillbot_upload_policy_info`.
- Route-to-policy coverage from `skillbot_upload_route_policy_info`.
- Runtime security toggles from `skillbot_upload_runtime_config`.
- Scanner availability from `skillbot_dependency_health{dependency="upload_malware_scanner"}`.
- Freshclam age from `skillbot_upload_freshclam_age_seconds`.
- Rejection and abuse trends from `skillbot_upload_security_rejections_total`.
- Quarantine/rejected/clean storage from `skillbot_upload_storage_bytes`.

Prometheus and exporter ports must not be public. Grafana must be authenticated and served over HTTPS.
```

---

## Task 10: Commit And Push Repo Changes

**Files:**
- All modified backend/grafana docs/config files

- [ ] **Step 1: Review diffs**

Run:

```powershell
git -C backend status --short
git -C backend diff --stat
git -C backend diff --check
```

If `grafana-main-app` is not inside a git repo, record that clearly and either:

```text
Keep it as direct deploy artifact only
```

or move the deployable monitoring stack into a tracked repo path in a follow-up plan.

- [ ] **Step 2: Run backend tests**

Run:

```powershell
cd backend
venv\Scripts\python -m pytest tests/test_upload_security_metrics_exporter.py tests/test_upload_policy_coverage.py tests/test_upload_deploy_validation.py -q
```

Expected:

```text
All selected tests pass.
```

- [ ] **Step 3: Commit backend changes**

Run:

```powershell
git -C backend add core/observability.py core/upload_security/metrics_exporter.py main_async.py tests/test_upload_security_metrics_exporter.py docs/UPLOAD_SECURITY.md docs/superpowers/plans/2026-06-21-jitsi-grafana-prometheus-monitoring.md
git -C backend commit -m "feat: export upload security monitoring config"
git -C backend push origin main
```

- [ ] **Step 4: Wait for backend CI/CD**

Run:

```powershell
gh run list --repo paritosh921/stoody-backend --branch main --limit 5
gh run watch --repo paritosh921/stoody-backend <run-id>
```

Expected:

```text
Production backend deploy passes.
```

---

## Task 11: Final Production Acceptance

- [ ] **Backend metrics acceptance**

Run from Jitsi server:

```bash
curl -fsS https://api.stoody.in/api/metrics | grep -E 'skillbot_upload_policy_limit|skillbot_upload_route_policy_info|skillbot_upload_runtime_config' | head -40
```

Expected:

```text
Effective upload policy/config metrics are present.
```

- [ ] **Monitoring stack acceptance**

Run on Jitsi server:

```bash
cd /opt/stoody-monitoring
docker compose ps
curl -fsS http://127.0.0.1:9090/-/ready
curl -fsS http://127.0.0.1:3000/api/health
```

Expected:

```text
All monitoring services healthy.
```

- [ ] **Security acceptance**

Run from local:

```powershell
curl.exe -I https://monitor.stoody.in
Test-NetConnection 3.6.147.238 -Port 9090
Test-NetConnection 3.6.147.238 -Port 9093
Test-NetConnection 3.6.147.238 -Port 9100
```

Expected:

```text
Grafana requires auth.
Prometheus, Alertmanager, and node exporter are not publicly reachable.
```

- [ ] **Capacity acceptance**

Run on Jitsi server:

```bash
free -h
df -h /
docker stats --no-stream
uptime
```

Expected:

```text
Disk below 70% used.
Memory available remains acceptable.
Jitsi containers remain healthy.
Monitoring does not materially increase load.
```

## Rollback Plan

If Jitsi performance degrades or disk pressure appears:

```bash
cd /opt/stoody-monitoring
docker compose down
sudo cp /etc/caddy/Caddyfile.bak.<timestamp> /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
df -h /
free -h
docker ps
```

If backend metrics auth/allowlist breaks Prometheus scraping:

```bash
sudo cp /etc/nginx/sites-enabled/<backend-site>.bak.<timestamp> /etc/nginx/sites-enabled/<backend-site>
sudo nginx -t
sudo systemctl reload nginx
```

## Open Decisions Before Execution

1. DNS name for Grafana:
   - Recommended: `monitor.stoody.in -> 3.6.147.238`
   - Temporary fallback: SSH tunnel to `127.0.0.1:3000`
2. Alert delivery channel:
   - Email, Slack, Telegram, or another receiver must be selected before Alertmanager can notify externally.
3. Git ownership of `grafana-main-app`:
   - It currently behaves like a local artifact, not a backend submodule/repo. Decide whether to keep it as direct deploy artifact or move it into a tracked repo path.
4. Backend metrics protection:
   - Recommended first step is Nginx allowlist for Jitsi IP.
   - Token-based metrics auth can be added later if IP allowlisting is too brittle.

## Self-Review

- Spec coverage:
  - Installs Grafana/Prometheus on the Jitsi EC2.
  - Includes file upload API limits and route policy mappings.
  - Includes runtime configuration and running statuses, not only usage stats.
  - Secures dashboard access with Caddy basic auth plus Grafana login.
  - Keeps sensitive exporter ports private.
  - Includes server capacity protection and rollback.
- Placeholder scan:
  - Remaining placeholders are intentional operator-provided secrets or DNS choices: `<strong-random-password>`, `<caddy-hashed-password>`, and backend Nginx site path.
- Type consistency:
  - Metric names use the `skillbot_upload_*` namespace already present in backend observability.
  - Upload policy source remains `backend/core/upload_security/policies.py`.
  - Route source remains `backend/core/upload_security/routes.py`.
