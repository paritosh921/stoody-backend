#!/usr/bin/env bash
# ============================================================================
# install-hub-software.sh — Install ExamPen Hub Python Packages
#
# Installs all hub Python packages into /opt/exampen/ with a dedicated
# virtual environment. Called during golden image build (inside chroot)
# or manually during development.
#
# Installed packages (from exam-conductor/hub/):
#   hub-common      — IPC definitions, shared config, data models
#   hub-supervisor  — Process manager, FSM orchestration, watchdog
#   hub-ble-mgr     — BLE dongle management (5 dongles × 8 pens)
#   hub-pen-sync    — GATT read, chunk transfer from pens
#   hub-timer       — Exam countdown, CLOCK_MONOTONIC, reboot recovery
#   hub-store       — Dual-write (SD+USB), fsync protocol
#   hub-uplink      — WiFi/mobile upload, resume ledger
#   hub-invig-ble   — Invigilator mobile BLE relay
#   hub-tui         — Textual-based terminal UI (8 screens)
#
# Usage:
#   sudo ./install-hub-software.sh [--from-source /path/to/hub]
# ============================================================================
set -euo pipefail

INSTALL_DIR="/opt/exampen"
VENV_DIR="${INSTALL_DIR}/venv"
BIN_DIR="${INSTALL_DIR}/bin"
HUB_SOURCE="${1:-}"

# ── Colour helpers ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[install]${NC} $*"; }
warn() { echo -e "${YELLOW}[install]${NC} $*"; }

# ── Hub packages in dependency order ───────────────────────────────────────
# hub-common must be installed first; all others depend on it.
# hub-supervisor is last — it imports from all other packages.
HUB_PACKAGES=(
    "hub-common"
    "hub-store"
    "hub-timer"
    "hub-ble-mgr"
    "hub-pen-sync"
    "hub-uplink"
    "hub-invig-ble"
    "hub-tui"
    "hub-supervisor"
)

# ── Create directory structure ─────────────────────────────────────────────
setup_dirs() {
    log "Creating directory structure..."
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$BIN_DIR"
    mkdir -p "${INSTALL_DIR}/etc"
    mkdir -p /etc/exampen
    mkdir -p /var/lib/exampen/data
    mkdir -p /var/lib/exampen/logs
    mkdir -p /var/log/exampen
    mkdir -p /mnt/exampen-backup
}

# ── Create virtual environment ─────────────────────────────────────────────
create_venv() {
    log "Creating Python 3.12 virtual environment at ${VENV_DIR}..."

    if [[ -d "$VENV_DIR" ]]; then
        warn "Existing venv found — removing and recreating."
        rm -rf "$VENV_DIR"
    fi

    python3.12 -m venv "$VENV_DIR"

    # Upgrade pip and install build tools
    "${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel

    log "Virtual environment created."
}

# ── Install hub packages ──────────────────────────────────────────────────
install_packages() {
    log "Installing hub packages..."

    if [[ -n "$HUB_SOURCE" && -d "$HUB_SOURCE" ]]; then
        # Install from local source (development or CI build)
        log "Installing from local source: ${HUB_SOURCE}"
        for pkg in "${HUB_PACKAGES[@]}"; do
            local pkg_dir="${HUB_SOURCE}/${pkg}"
            if [[ -d "$pkg_dir" ]]; then
                log "  Installing ${pkg} from ${pkg_dir}..."
                "${VENV_DIR}/bin/pip" install --no-cache-dir "${pkg_dir}"
            else
                warn "  Package directory not found: ${pkg_dir} — skipping"
            fi
        done
    else
        # Install from PyPI or pre-built wheels (production build)
        # In production, wheels are placed in /tmp/hub-wheels/ by the
        # build pipeline before this script runs.
        local wheel_dir="/tmp/hub-wheels"
        if [[ -d "$wheel_dir" ]]; then
            log "Installing from pre-built wheels in ${wheel_dir}..."
            "${VENV_DIR}/bin/pip" install --no-cache-dir --find-links "$wheel_dir" \
                "${HUB_PACKAGES[@]}"
        else
            # Fallback: install from PyPI (packages must be published)
            log "Installing from PyPI..."
            "${VENV_DIR}/bin/pip" install --no-cache-dir "${HUB_PACKAGES[@]}"
        fi
    fi

    log "Hub packages installed."
}

# ── Verify installation ───────────────────────────────────────────────────
verify_install() {
    log "Verifying installation..."

    # Check that key entry points are importable
    local failed=0
    for pkg in "${HUB_PACKAGES[@]}"; do
        # Convert package name to Python module name (hub-common → hub_common)
        local module_name="${pkg//-/_}"
        if "${VENV_DIR}/bin/python" -c "import ${module_name}" 2>/dev/null; then
            log "  ${pkg} (${module_name}) — OK"
        else
            warn "  ${pkg} (${module_name}) — import failed (may not be installed yet)"
            failed=$((failed + 1))
        fi
    done

    if [[ $failed -gt 0 ]]; then
        warn "${failed} package(s) could not be verified. This is expected during"
        warn "golden image build if packages are not yet published."
    fi
}

# ── Create hub-supervisor wrapper script ──────────────────────────────────
create_wrapper() {
    log "Creating hub-supervisor wrapper at ${BIN_DIR}/hub-supervisor..."

    cat > "${BIN_DIR}/hub-supervisor" <<'WRAPPER'
#!/usr/bin/env bash
# ============================================================================
# hub-supervisor — ExamPen Hub Supervisor Launcher
#
# Wrapper script that activates the Python venv and launches the
# hub-supervisor process. Called by systemd via exampen-supervisor.service.
#
# Environment variables (set by systemd):
#   EXAMPEN_DATA    — Path to primary data directory (/var/lib/exampen)
#   EXAMPEN_BACKUP  — Path to USB backup mount (/mnt/exampen-backup)
#
# Additional environment variables (optional):
#   EXAMPEN_LOG_LEVEL — Logging level (DEBUG, INFO, WARNING, ERROR)
#   EXAMPEN_CONFIG    — Path to hub config (/etc/exampen/hub.conf)
# ============================================================================
set -euo pipefail

# Activate the ExamPen virtual environment
VENV_DIR="/opt/exampen/venv"
export PATH="${VENV_DIR}/bin:${PATH}"
export VIRTUAL_ENV="${VENV_DIR}"

# Set defaults for environment variables if not already set by systemd
export EXAMPEN_DATA="${EXAMPEN_DATA:-/var/lib/exampen}"
export EXAMPEN_BACKUP="${EXAMPEN_BACKUP:-/mnt/exampen-backup}"
export EXAMPEN_CONFIG="${EXAMPEN_CONFIG:-/etc/exampen/hub.conf}"
export EXAMPEN_LOG_LEVEL="${EXAMPEN_LOG_LEVEL:-INFO}"

# Launch the supervisor — exec replaces this shell so systemd tracks
# the Python process directly (important for Type=notify and watchdog)
exec "${VENV_DIR}/bin/python" -m hub_supervisor "$@"
WRAPPER

    chmod +x "${BIN_DIR}/hub-supervisor"
    log "Wrapper script created."
}

# ── Set ownership and permissions ──────────────────────────────────────────
set_permissions() {
    log "Setting ownership and permissions..."

    # The hub software runs as root (required for BLE and USB access)
    # In future, a dedicated exampen user with appropriate capabilities
    # could be used for privilege separation.
    chmod -R 755 "$INSTALL_DIR"
    chmod 755 "${BIN_DIR}/hub-supervisor"

    # Data directories need write access
    chmod 755 /var/lib/exampen
    chmod 755 /var/lib/exampen/data
    chmod 755 /var/lib/exampen/logs
    chmod 755 /var/log/exampen

    log "Permissions set."
}

# ── Main ──────────────────────────────────────────────────────────────────
main() {
    log "============================================"
    log "ExamPen Hub Software Installer"
    log "============================================"

    # Parse optional --from-source argument
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --from-source) HUB_SOURCE="$2"; shift 2 ;;
            *)             shift ;;
        esac
    done

    setup_dirs
    create_venv
    install_packages
    verify_install
    create_wrapper
    set_permissions

    log "============================================"
    log "Installation complete!"
    log "  Venv:    ${VENV_DIR}"
    log "  Binary:  ${BIN_DIR}/hub-supervisor"
    log "  Data:    /var/lib/exampen"
    log "  Config:  /etc/exampen"
    log "  Logs:    /var/log/exampen"
    log "============================================"
}

main "$@"
