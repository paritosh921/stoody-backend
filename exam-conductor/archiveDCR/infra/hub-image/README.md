# ExamPen Hub Golden Image Builder

Build scripts for creating the ExamPen hub Raspberry Pi golden image. The
output is a compressed `.img.xz` file that can be flashed directly to an
SD card for RPi 4B or RPi 5 deployment.

## Prerequisites

### Host System

- Linux (Ubuntu 22.04+ recommended) — x86_64 or arm64
- Root access (sudo)

### Required Packages

```bash
# Debian/Ubuntu
sudo apt-get install \
    qemu-user-static binfmt-support \
    kpartx dosfstools e2fsprogs \
    xz-utils wget rsync \
    fdisk util-linux
```

On arm64 hosts (e.g., building on another RPi), `qemu-user-static` is not
needed since native chroot works directly.

### Disk Space

The build requires approximately 40 GB of free disk space:
- ~4 GB for the downloaded Ubuntu base image
- ~26 GB for the uncompressed output image
- ~8 GB working space

## Usage

### Build the Image

```bash
sudo ./build-image.sh --version 1.0.0
```

Options:
- `--version <semver>` — Version tag for the output image (default: `1.0.0`)
- `--skip-download` — Reuse a previously downloaded base image

The build process takes 15–30 minutes depending on network speed and host
performance.

### Output

```
output/
├── exampen-hub-1.0.0.img.xz         # Compressed golden image
├── exampen-hub-1.0.0.img.xz.sha256  # SHA-256 checksum
└── exampen-hub-1.0.0.img.xz.md5     # MD5 checksum
```

### Flash to SD Card

```bash
# Identify your SD card device (e.g., /dev/sdb)
lsblk

# Flash (replace /dev/sdX with your actual device)
xzcat output/exampen-hub-1.0.0.img.xz | sudo dd of=/dev/sdX bs=4M status=progress
sync
```

**Warning:** Double-check the target device. `dd` will overwrite without
confirmation.

## Image Contents

### Operating System

| Parameter   | Value                                      |
|-------------|--------------------------------------------|
| OS          | Ubuntu Server 24.04 LTS (Noble Numbat)     |
| Arch        | arm64                                      |
| Kernel      | linux-raspi (Ubuntu-maintained)            |
| Init        | systemd                                    |
| Locale      | en_US.UTF-8                                |
| Timezone    | UTC                                        |
| WiFi Region | US (locked — not configurable)             |

### Partition Layout

| # | Mount             | Size   | Filesystem | Purpose                |
|---|-------------------|--------|------------|------------------------|
| 1 | /boot/firmware    | 512 MB | FAT32      | RPi bootloader + kernel|
| 2 | /                 | 8 GB   | ext4       | OS + hub software      |
| 3 | /var/lib/exampen  | 16 GB  | ext4       | Pen data primary store |
| 4 | (swap)            | 1 GB   | swap       | OOM safety             |

USB backup drive mounts at `/mnt/exampen-backup` with `nofail`.

Minimum SD card size: 32 GB.

### Pre-installed Packages

| Package            | Purpose                           |
|--------------------|-----------------------------------|
| bluez (5.72+)      | Bluetooth Low Energy stack        |
| bluez-tools        | BLE utilities                     |
| python3.12         | Hub software runtime              |
| python3-pip        | Package installer                 |
| python3-venv       | Virtual environment support       |
| sqlite3            | Local database engine             |
| NetworkManager     | WiFi management (nmcli)           |
| chrony             | NTP time synchronization          |
| openssh-server     | Remote debug access               |
| htop, iotop        | Performance monitoring            |
| lsof, strace       | Debugging tools                   |
| usbutils           | USB device enumeration (lsusb)    |

### ExamPen Hub Software

Installed at `/opt/exampen/` with a Python 3.12 venv at `/opt/exampen/venv/`.

| Package          | Role                                          |
|------------------|-----------------------------------------------|
| hub-common       | IPC definitions, shared config, data models   |
| hub-supervisor   | Process manager, FSM orchestration, watchdog  |
| hub-ble-mgr      | BLE dongle management (5 dongles x 8 pens)   |
| hub-pen-sync     | GATT read, chunk transfer from pens           |
| hub-timer        | Exam countdown, reboot recovery               |
| hub-store        | Dual-write (SD+USB), fsync protocol           |
| hub-uplink       | WiFi/mobile upload, resume ledger             |
| hub-invig-ble    | Invigilator mobile BLE relay                  |
| hub-tui          | Textual-based terminal UI                     |

### systemd Services

The main service is `exampen-supervisor.service` (Type=notify, WatchdogSec=30).
It manages all child hub processes internally.

### Directory Layout

```
/etc/exampen/            — Configuration files
/var/lib/exampen/        — Primary data store (SQLite DB + pen data)
/var/log/exampen/        — Application logs
/opt/exampen/            — Hub software installation
/opt/exampen/venv/       — Python virtual environment
/opt/exampen/bin/        — Executable wrappers
/mnt/exampen-backup/     — USB backup drive mount point
```

## File Reference

| File                              | Purpose                                |
|-----------------------------------|----------------------------------------|
| `build-image.sh`                  | Main image build orchestrator          |
| `partition-layout.sfdisk`         | SD card partition table (sfdisk format)|
| `install-hub-software.sh`         | Hub Python package installer           |
| `systemd/exampen-supervisor.service` | systemd unit file                   |
| `systemd/exampen-bluetooth.conf`  | BlueZ configuration overrides          |
| `config/crda.conf`                | WiFi regulatory domain (US)            |
| `config/cfg80211.conf`            | Kernel WiFi regulatory module option   |
| `config/fstab.append`             | USB backup mount fstab entry           |
| `config/chrony.conf`              | NTP client configuration               |
| `CHECKSUMS.md`                    | Image verification checksums template  |

## First-Boot Sequence

After flashing the image and powering on the RPi:

1. systemd boots and starts `exampen-supervisor`
2. Supervisor detects first-boot (no `/etc/exampen/hub.conf`)
3. TUI launches on HDMI/serial and forces the Setup Screen
4. Operator enters: hub unique code, backend URL, WiFi credentials
5. Hub connects to WiFi, verifies backend, sends provisioning request
6. Backend responds with hub ID, institute info, invigilator codes, pen inventory
7. Hub stores config and transitions to PROVISIONED state

See `HUB_DEPLOYMENT_SPEC.md` section 7 for full first-boot specification.

## Development

### Building from Local Source

To install hub packages from local source during development:

```bash
sudo ./install-hub-software.sh --from-source /path/to/exam-conductor/hub
```

### Testing Individual Components

```bash
# Activate the hub venv
source /opt/exampen/venv/bin/activate

# Run a specific module
python -m hub_supervisor
python -m hub_tui
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| BlueZ version too old | Ubuntu repos outdated | Build BlueZ from source or use PPA |
| qemu-user-static missing | Cross-compilation on x86_64 | `apt install qemu-user-static binfmt-support` |
| Partition mount fails | Loop device busy | `losetup -D` to detach all loops |
| Image too large for SD card | Card < 32 GB | Use 32 GB or larger SD card |
| WiFi channels restricted | Wrong regulatory domain | Verify `/etc/default/crda` is `REGDOMAIN=US` |
