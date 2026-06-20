from pathlib import Path

from scripts.validate_upload_security_deploy import run_upload_security_deploy_validation


class FakeRunner:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, command, timeout=10):
        key = tuple(command)
        self.calls.append(key)
        if len(command) == 3 and tuple(command[:2]) == ("clamdscan", "--fdpass"):
            key = ("clamdscan", "--fdpass", "<path>")
        return self.responses.get(key, (0, "OK", ""))


def test_deploy_validation_passes_with_required_runtime_controls(tmp_path):
    socket = tmp_path / "clamd.ctl"
    socket.write_text("socket", encoding="utf-8")
    socket.chmod(0o660)
    private_root = tmp_path / "uploads"
    for prefix in ("quarantine", "rejected", "clean"):
        (private_root / prefix).mkdir(parents=True)
    runner = FakeRunner(
        {
            ("systemctl", "is-active", "clamav-daemon"): (0, "active\n", ""),
            ("systemctl", "is-active", "clamav-freshclam"): (0, "active\n", ""),
            ("clamdscan", "--fdpass", "<path>"): (1, "Eicar-Test-Signature FOUND\n", ""),
            ("curl", "-fsS", "https://api.example.test/health"): (
                0,
                '{"upload_malware_scanner":{"available":true}}',
                "",
            ),
        }
    )

    result = run_upload_security_deploy_validation(
        env={
            "UPLOAD_SCAN_REQUIRED": "true",
            "UPLOAD_AV_FAIL_CLOSED": "true",
            "CLAMAV_SOCKET": str(socket),
            "CLAMAV_SOCKET_OWNER": "clamav",
            "CLAMAV_SOCKET_GROUP": "clamav",
            "CLAMAV_SOCKET_MODE": "660",
            "BACKEND_SERVICE_USER": "ubuntu",
            "UPLOAD_PRIVATE_LOCAL_DIR": str(private_root),
            "BACKEND_HEALTH_URL": "https://api.example.test/health",
        },
        runner=runner,
        nginx_config_paths=[],
    )

    assert result.ok is True
    assert result.failed_checks == []


def test_deploy_validation_fails_when_scanner_controls_are_disabled(tmp_path):
    result = run_upload_security_deploy_validation(
        env={
            "UPLOAD_SCAN_REQUIRED": "false",
            "UPLOAD_AV_FAIL_CLOSED": "false",
            "CLAMAV_SOCKET": str(tmp_path / "missing.sock"),
            "UPLOAD_PRIVATE_LOCAL_DIR": str(tmp_path / "uploads"),
        },
        runner=FakeRunner({}),
        nginx_config_paths=[],
    )

    assert result.ok is False
    assert "UPLOAD_SCAN_REQUIRED" in result.failed_checks
    assert "UPLOAD_AV_FAIL_CLOSED" in result.failed_checks
    assert "CLAMAV_SOCKET_EXISTS" in result.failed_checks


def test_deploy_validation_fails_if_nginx_serves_private_upload_root(tmp_path):
    private_root = tmp_path / "uploads"
    private_root.mkdir()
    nginx_conf = tmp_path / "nginx.conf"
    nginx_conf.write_text(f"location /uploads {{ alias {private_root}; }}", encoding="utf-8")

    result = run_upload_security_deploy_validation(
        env={
            "UPLOAD_SCAN_REQUIRED": "true",
            "UPLOAD_AV_FAIL_CLOSED": "true",
            "CLAMAV_SOCKET": str(tmp_path / "missing.sock"),
            "UPLOAD_PRIVATE_LOCAL_DIR": str(private_root),
        },
        runner=FakeRunner({}),
        nginx_config_paths=[Path(nginx_conf)],
    )

    assert "NGINX_PRIVATE_UPLOAD_NOT_SERVED" in result.failed_checks


def test_deploy_validation_enforces_exact_clamav_socket_hardening_checks():
    source = Path("scripts/validate_upload_security_deploy.py").read_text(encoding="utf-8")

    assert "CLAMAV_SOCKET_MODE_660" in source
    assert "CLAMAV_SOCKET_OWNER_GROUP" in source
    assert "BACKEND_SERVICE_USER_IN_CLAMAV_GROUP" in source


def test_remote_deploy_script_hardens_clamav_socket_before_backend_restart():
    source = Path("ops/remote_deploy_python_service.sh").read_text(encoding="utf-8")

    assert "ensure_clamav_socket_hardening" in source
    assert "LocalSocketMode" in source
    assert "LocalSocketGroup" in source
    assert "clamav-daemon.socket.d/upload-security.conf" in source
    assert "SocketMode=%s" in source
    assert "sudo systemctl stop clamav-daemon.service clamav-daemon.socket" in source
    assert "sudo systemctl start clamav-daemon.socket" in source
    assert "sudo usermod -aG" in source
    assert 'sudo -u "$upload_owner" clamdscan --fdpass /etc/hosts' in source
    assert source.index("ensure_clamav_socket_hardening") < source.index("sudo systemctl restart clamav-daemon")
    assert source.rindex("ensure_clamav_socket_hardening") < source.rindex("ensure_private_upload_dirs")


def test_remote_deploy_script_serializes_full_host_deploy():
    source = Path("ops/remote_deploy_python_service.sh").read_text(encoding="utf-8")

    assert "STOODY_DEPLOY_LOCK_PATH" in source
    assert "STOODY_DEPLOY_LOCK_TIMEOUT_SECONDS" in source
    assert "flock -w" in source
    assert "acquire_deploy_lock" in source
    assert source.rindex("acquire_deploy_lock") < source.index('sync_git_repo "$APP_PATH" "$BRANCH"')


def test_backend_deploy_workflows_queue_instead_of_canceling_remote_deploys():
    for workflow in (
        Path(".github/workflows/deploy-prod-backend.yml"),
        Path(".github/workflows/deploy-dev-backend.yml"),
    ):
        source = workflow.read_text(encoding="utf-8")
        assert "concurrency:" in source
        assert "cancel-in-progress: false" in source
