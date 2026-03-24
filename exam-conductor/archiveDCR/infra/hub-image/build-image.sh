#!/usr/bin/env bash
# ============================================================================
# build-image.sh — ExamPen Hub Golden Image Builder
#
# Downloads Ubuntu Server 24.04 LTS arm64 RPi base image and customizes it
# into a ready-to-flash golden image for ExamPen hub deployment.
#
# Requirements:
#   - Linux host (x86_64 or arm64)
#   - qemu-user-static (for cross-arch chroot on x86_64 hosts)
#   - losetup, kpartx, sfdisk, mkfs.ext4, mkfs.vfat, mkswap
#   - xz-utils, wget, rsync
#   - Root privileges (sudo)
#
# Usage:
#   sudo ./build-image.sh [--version 1.0.0] [--skip-download]
# ============================================================================
set -euo pipefail

# ── Configurable parameters ────────────────────────────────────────────────
EXAMPEN_VERSION="${1:-1.0.0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-false}"
UBUNTU_VERSION="24.04"
UBUNTU_IMAGE_URL="https://cdimage.ubuntu.com/releases/${UBUNTU_VERSION}/release/ubuntu-${UBUNTU_VERSION}-preinstalled-server-arm64+raspi.img.xz"
UBUNTU_IMAGE_CHECKSUM_URL="${UBUNTU_IMAGE_URL}.sha256"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/build"
MOUNT_DIR="${WORK_DIR}/mnt"
CHROOT_DIR="${WORK_DIR}/chroot"
OUTPUT_DIR="${SCRIPT_DIR}/output"
DOWNLOAD_DIR="${SCRIPT_DIR}/downloads"

# Image geometry — matches HUB_DEPLOYMENT_SPEC §1.4
BOOT_SIZE_MB=512
ROOTFS_SIZE_MB=8192     # 8 GB
DATA_SIZE_MB=16384      # 16 GB
SWAP_SIZE_MB=1024       # 1 GB
TOTAL_SIZE_MB=$(( BOOT_SIZE_MB + ROOTFS_SIZE_MB + DATA_SIZE_MB + SWAP_SIZE_MB + 4 ))

OUTPUT_IMAGE="exampen-hub-${EXAMPEN_VERSION}.img"
OUTPUT_COMPRESSED="exampen-hub-${EXAMPEN_VERSION}.img.xz"

# ── Colour helpers ─────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[build]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[error]${NC} $*" >&2; }

# ── Pre-flight checks ─────────────────────────────────────────────────────
preflight() {
    log "Running pre-flight checks..."

    if [[ $EUID -ne 0 ]]; then
        err "This script must be run as root (sudo)."
        exit 1
    fi

    local required_cmds=(losetup kpartx sfdisk mkfs.ext4 mkfs.vfat mkswap xz wget rsync)
    for cmd in "${required_cmds[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            err "Required command not found: $cmd"
            exit 1
        fi
    done

    # Check for qemu-user-static on x86_64 hosts (needed for arm64 chroot)
    local host_arch
    host_arch="$(uname -m)"
    if [[ "$host_arch" == "x86_64" ]]; then
        if ! command -v qemu-aarch64-static &>/dev/null; then
            err "qemu-user-static required for cross-arch chroot on x86_64 host."
            err "Install with: apt-get install qemu-user-static binfmt-support"
            exit 1
        fi
        log "Cross-compilation mode: x86_64 host → arm64 target (qemu-user-static)"
    else
        log "Native build: arm64 host → arm64 target"
    fi

    log "Pre-flight checks passed."
}

# ── Download base image ────────────────────────────────────────────────────
download_base_image() {
    mkdir -p "$DOWNLOAD_DIR"
    local base_xz="${DOWNLOAD_DIR}/ubuntu-${UBUNTU_VERSION}-arm64-raspi.img.xz"
    local base_img="${DOWNLOAD_DIR}/ubuntu-${UBUNTU_VERSION}-arm64-raspi.img"

    if [[ "$SKIP_DOWNLOAD" == "true" && -f "$base_img" ]]; then
        log "Skipping download — using cached base image."
        echo "$base_img"
        return
    fi

    log "Downloading Ubuntu Server ${UBUNTU_VERSION} LTS arm64 RPi image..."
    wget -q --show-progress -O "$base_xz" "$UBUNTU_IMAGE_URL"

    log "Downloading checksum..."
    wget -q -O "${base_xz}.sha256" "$UBUNTU_IMAGE_CHECKSUM_URL"

    log "Verifying checksum..."
    (cd "$DOWNLOAD_DIR" && sha256sum -c "${base_xz}.sha256")

    log "Decompressing base image..."
    xz -dk "$base_xz"

    echo "$base_img"
}

# ── Create output image with correct partition layout ──────────────────────
create_image() {
    log "Creating blank image: ${TOTAL_SIZE_MB} MB..."
    mkdir -p "$WORK_DIR" "$OUTPUT_DIR"

    dd if=/dev/zero of="${WORK_DIR}/${OUTPUT_IMAGE}" bs=1M count="$TOTAL_SIZE_MB" status=progress

    log "Applying partition layout from partition-layout.sfdisk..."
    sfdisk "${WORK_DIR}/${OUTPUT_IMAGE}" < "${SCRIPT_DIR}/partition-layout.sfdisk"

    log "Partition layout applied."
}

# ── Mount image partitions ─────────────────────────────────────────────────
mount_image() {
    log "Setting up loop device and mapping partitions..."
    LOOP_DEV=$(losetup --find --show --partscan "${WORK_DIR}/${OUTPUT_IMAGE}")
    log "Loop device: ${LOOP_DEV}"

    # Wait for partition devices to appear
    sleep 1
    kpartx -av "$LOOP_DEV"
    sleep 1

    local mapper_base
    mapper_base="/dev/mapper/$(basename "$LOOP_DEV")"

    PART_BOOT="${mapper_base}p1"
    PART_ROOTFS="${mapper_base}p2"
    PART_DATA="${mapper_base}p3"
    PART_SWAP="${mapper_base}p4"

    log "Formatting partitions..."
    mkfs.vfat -F 32 -n "BOOT" "$PART_BOOT"
    mkfs.ext4 -L "rootfs" "$PART_ROOTFS"
    mkfs.ext4 -L "exampen-data" -O "^has_journal" "$PART_DATA"
    mkswap -L "swap" "$PART_SWAP"

    # Confirm data partition label for fstab LABEL= mount at boot
    e2label "$PART_DATA" "exampen-data"

    # Enable noatime on data partition via tune2fs
    tune2fs -o journal_data_writeback "$PART_DATA" 2>/dev/null || true

    log "Mounting partitions..."
    mkdir -p "${CHROOT_DIR}"
    mount "$PART_ROOTFS" "$CHROOT_DIR"
    mkdir -p "${CHROOT_DIR}/boot/firmware"
    mount "$PART_BOOT" "${CHROOT_DIR}/boot/firmware"
    mkdir -p "${CHROOT_DIR}/var/lib/exampen"
    mount "$PART_DATA" "${CHROOT_DIR}/var/lib/exampen"

    log "Partitions mounted at ${CHROOT_DIR}"
}

# ── Copy base image contents ──────────────────────────────────────────────
copy_base() {
    local base_img="$1"

    log "Mounting base Ubuntu image to extract contents..."
    local base_loop
    base_loop=$(losetup --find --show --partscan "$base_img")
    sleep 1
    kpartx -av "$base_loop"
    sleep 1

    local base_mapper
    base_mapper="/dev/mapper/$(basename "$base_loop")"

    mkdir -p "${MOUNT_DIR}/base-boot" "${MOUNT_DIR}/base-rootfs"
    mount "${base_mapper}p1" "${MOUNT_DIR}/base-boot"
    mount "${base_mapper}p2" "${MOUNT_DIR}/base-rootfs"

    log "Copying boot partition..."
    rsync -aHAX "${MOUNT_DIR}/base-boot/" "${CHROOT_DIR}/boot/firmware/"

    log "Copying rootfs..."
    rsync -aHAX "${MOUNT_DIR}/base-rootfs/" "${CHROOT_DIR}/"

    log "Unmounting base image..."
    umount "${MOUNT_DIR}/base-boot"
    umount "${MOUNT_DIR}/base-rootfs"
    kpartx -dv "$base_loop"
    losetup -d "$base_loop"
    rmdir "${MOUNT_DIR}/base-boot" "${MOUNT_DIR}/base-rootfs" 2>/dev/null || true

    log "Base image contents copied."
}

# ── Chroot customization ──────────────────────────────────────────────────
setup_chroot() {
    log "Preparing chroot environment..."

    # Bind-mount host resources for chroot
    mount --bind /dev  "${CHROOT_DIR}/dev"
    mount --bind /proc "${CHROOT_DIR}/proc"
    mount --bind /sys  "${CHROOT_DIR}/sys"
    mount --bind /dev/pts "${CHROOT_DIR}/dev/pts"

    # Copy DNS resolution into chroot
    cp /etc/resolv.conf "${CHROOT_DIR}/etc/resolv.conf"

    # Copy qemu static binary if cross-compiling
    if [[ "$(uname -m)" == "x86_64" ]]; then
        cp /usr/bin/qemu-aarch64-static "${CHROOT_DIR}/usr/bin/"
    fi
}

customize_image() {
    log "Customizing image inside chroot..."

    chroot "$CHROOT_DIR" /bin/bash -e <<'CHROOT_SCRIPT'
# ── Locale & timezone ──────────────────────────────────────────────────────
echo "en_US.UTF-8 UTF-8" > /etc/locale.gen
locale-gen
update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
ln -sf /usr/share/zoneinfo/UTC /etc/localtime
echo "UTC" > /etc/timezone
dpkg-reconfigure -f noninteractive tzdata

# ── WiFi regulatory domain (US — locked, not configurable) ────────────────
echo 'REGDOMAIN=US' > /etc/default/crda
mkdir -p /etc/modprobe.d
echo 'options cfg80211 ieee80211_regdom=US' > /etc/modprobe.d/cfg80211.conf

# ── Update and install packages ────────────────────────────────────────────
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get upgrade -y -qq

apt-get install -y -qq --no-install-recommends \
    bluez \
    bluez-tools \
    python3.12 \
    python3-pip \
    python3-venv \
    sqlite3 \
    network-manager \
    chrony \
    htop \
    iotop \
    lsof \
    strace \
    usbutils \
    openssh-server \
    wireless-regdb \
    crda

# ── Verify BlueZ version ≥ 5.72 ──────────────────────────────────────────
BLUEZ_VER=$(dpkg -s bluez | grep '^Version:' | awk '{print $2}' | cut -d- -f1)
REQUIRED_VER="5.72"
if dpkg --compare-versions "$BLUEZ_VER" lt "$REQUIRED_VER"; then
    echo "ERROR: BlueZ version $BLUEZ_VER < $REQUIRED_VER required"
    exit 1
fi
echo "BlueZ version $BLUEZ_VER OK (>= $REQUIRED_VER)"

# ── Create ExamPen directories ─────────────────────────────────────────────
mkdir -p /etc/exampen
mkdir -p /var/lib/exampen/data
mkdir -p /var/lib/exampen/logs
mkdir -p /var/log/exampen
mkdir -p /mnt/exampen-backup
mkdir -p /opt/exampen/bin

# ── Disable cloud-init (golden image, not cloud-init provisioned) ──────────
if dpkg -l cloud-init &>/dev/null; then
    touch /etc/cloud/cloud-init.disabled
    # Optionally remove cloud-init entirely to save space
    apt-get purge -y -qq cloud-init
    rm -rf /etc/cloud /var/lib/cloud
fi

# ── Configure chrony for NTP ──────────────────────────────────────────────
# The main chrony.conf is deployed via config file; ensure service is enabled
systemctl enable chrony

# ── Enable NetworkManager, disable systemd-networkd ───────────────────────
systemctl enable NetworkManager
systemctl disable systemd-networkd 2>/dev/null || true
systemctl disable systemd-resolved 2>/dev/null || true

# ── Enable bluetooth ──────────────────────────────────────────────────────
systemctl enable bluetooth

# ── Enable SSH (for remote debug access) ──────────────────────────────────
systemctl enable ssh

# ── Clean up ──────────────────────────────────────────────────────────────
apt-get autoremove -y -qq
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf /tmp/* /var/tmp/*

echo "Chroot customization complete."
CHROOT_SCRIPT

    log "Chroot customization finished."
}

# ── Install config files ──────────────────────────────────────────────────
install_configs() {
    log "Installing configuration files..."

    # WiFi regulatory domain
    cp "${SCRIPT_DIR}/config/crda.conf" "${CHROOT_DIR}/etc/default/crda"
    cp "${SCRIPT_DIR}/config/cfg80211.conf" "${CHROOT_DIR}/etc/modprobe.d/cfg80211.conf"

    # Chrony NTP config
    cp "${SCRIPT_DIR}/config/chrony.conf" "${CHROOT_DIR}/etc/chrony/chrony.conf"

    # Append USB mount entry to fstab
    cat "${SCRIPT_DIR}/config/fstab.append" >> "${CHROOT_DIR}/etc/fstab"

    # BlueZ config overrides
    mkdir -p "${CHROOT_DIR}/etc/bluetooth"
    cp "${SCRIPT_DIR}/systemd/exampen-bluetooth.conf" "${CHROOT_DIR}/etc/bluetooth/exampen.conf"

    # systemd service files
    cp "${SCRIPT_DIR}/systemd/exampen-supervisor.service" \
       "${CHROOT_DIR}/etc/systemd/system/exampen-supervisor.service"

    # Enable the supervisor service
    chroot "$CHROOT_DIR" systemctl enable exampen-supervisor.service

    log "Configuration files installed."
}

# ── Install hub software ──────────────────────────────────────────────────
install_hub_software() {
    log "Installing ExamPen hub software..."
    cp "${SCRIPT_DIR}/install-hub-software.sh" "${CHROOT_DIR}/tmp/install-hub-software.sh"
    chmod +x "${CHROOT_DIR}/tmp/install-hub-software.sh"
    chroot "$CHROOT_DIR" /tmp/install-hub-software.sh
    rm -f "${CHROOT_DIR}/tmp/install-hub-software.sh"
    log "Hub software installed."
}

# ── Teardown chroot ───────────────────────────────────────────────────────
teardown_chroot() {
    log "Tearing down chroot..."
    umount "${CHROOT_DIR}/dev/pts"  2>/dev/null || true
    umount "${CHROOT_DIR}/dev"      2>/dev/null || true
    umount "${CHROOT_DIR}/proc"     2>/dev/null || true
    umount "${CHROOT_DIR}/sys"      2>/dev/null || true

    # Remove qemu binary if we copied it
    rm -f "${CHROOT_DIR}/usr/bin/qemu-aarch64-static"

    # Restore original resolv.conf
    rm -f "${CHROOT_DIR}/etc/resolv.conf"
    chroot "$CHROOT_DIR" ln -sf /run/systemd/resolve/resolv.conf /etc/resolv.conf 2>/dev/null || true
}

# ── Unmount and compress ──────────────────────────────────────────────────
finalize_image() {
    log "Syncing filesystems..."
    sync

    log "Unmounting partitions..."
    umount "${CHROOT_DIR}/boot/firmware" 2>/dev/null || true
    umount "${CHROOT_DIR}/var/lib/exampen" 2>/dev/null || true
    umount "${CHROOT_DIR}" 2>/dev/null || true

    log "Removing device mappings..."
    kpartx -dv "$LOOP_DEV"
    losetup -d "$LOOP_DEV"

    log "Compressing image with xz (this may take a while)..."
    xz -9 -T0 --verbose "${WORK_DIR}/${OUTPUT_IMAGE}"

    mv "${WORK_DIR}/${OUTPUT_IMAGE}.xz" "${OUTPUT_DIR}/${OUTPUT_COMPRESSED}"

    log "Generating checksums..."
    (cd "$OUTPUT_DIR" && sha256sum "$OUTPUT_COMPRESSED" > "${OUTPUT_COMPRESSED}.sha256")
    (cd "$OUTPUT_DIR" && md5sum "$OUTPUT_COMPRESSED" > "${OUTPUT_COMPRESSED}.md5")

    log "Image build complete!"
    log "Output: ${OUTPUT_DIR}/${OUTPUT_COMPRESSED}"
    log "SHA256: $(cat "${OUTPUT_DIR}/${OUTPUT_COMPRESSED}.sha256")"
}

# ── Cleanup on failure ────────────────────────────────────────────────────
cleanup() {
    warn "Cleaning up after error..."
    teardown_chroot 2>/dev/null || true
    umount "${CHROOT_DIR}/boot/firmware" 2>/dev/null || true
    umount "${CHROOT_DIR}/var/lib/exampen" 2>/dev/null || true
    umount "${CHROOT_DIR}" 2>/dev/null || true
    if [[ -n "${LOOP_DEV:-}" ]]; then
        kpartx -dv "$LOOP_DEV" 2>/dev/null || true
        losetup -d "$LOOP_DEV" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Main ──────────────────────────────────────────────────────────────────
main() {
    log "============================================"
    log "ExamPen Hub Golden Image Builder"
    log "Version: ${EXAMPEN_VERSION}"
    log "============================================"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --version)  EXAMPEN_VERSION="$2"; shift 2 ;;
            --skip-download) SKIP_DOWNLOAD="true"; shift ;;
            *)          shift ;;
        esac
    done

    preflight

    local base_img
    base_img=$(download_base_image)

    create_image
    mount_image
    copy_base "$base_img"
    setup_chroot
    customize_image
    install_configs
    install_hub_software
    teardown_chroot
    finalize_image

    # Disable the trap since we succeeded
    trap - EXIT

    log "============================================"
    log "Build successful!"
    log "Flash with:  xzcat ${OUTPUT_DIR}/${OUTPUT_COMPRESSED} | sudo dd of=/dev/sdX bs=4M status=progress"
    log "============================================"
}

main "$@"
