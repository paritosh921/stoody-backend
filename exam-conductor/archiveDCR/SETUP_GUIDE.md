# ExamPen -- Step-by-Step Setup Guide

This guide walks you through setting up the entire ExamPen system from scratch: development machine, Docker infrastructure, Raspberry Pi hub, BLE pens, and running a full end-to-end exam. Written for developers who may have never touched an RPi before.

---

## Part 0: Prerequisites & Hardware

### Hardware You Need

| Item | Specification | Purpose |
|------|--------------|---------|
| Development PC | Windows 11 / macOS / Linux, 16 GB+ RAM | Run backend services, frontends, Docker |
| Raspberry Pi 4B or 5 | 4 GB+ RAM | ExamPen hub |
| Micro SD card | 32 GB+ (Class 10 / U1 minimum) | RPi boot + OS + primary data |
| USB thumb drive | 16 GB+, ext4-formatted | Secondary data store (dual-write) |
| USB BLE dongles | 5x (any CSR8510 / Realtek-based) | BLE communication with pens |
| Powered USB hub | 7+ port, externally powered | Power the 5 BLE dongles reliably |
| Ethernet cable | Cat5e or better, 1-2 meters | Direct PC-to-RPi connection (headless setup) |
| USB-C power supply | 5V 3A (RPi 4B) or 5V 5A (RPi 5) | Power the RPi |
| BLE pens (P05 model) | As many as needed (up to 40 per hub) | Capture handwritten answers |
| nRF52840-DK (optional) | Nordic dev kit | Pen simulator if no physical pens |

### Software You Need on Your Dev PC

| Software | Version | Download |
|----------|---------|----------|
| Docker Desktop | 24+ with Compose V2 | https://www.docker.com/products/docker-desktop/ |
| Python | 3.12+ | https://www.python.org/downloads/ |
| Node.js | 22+ (LTS) | https://nodejs.org/ |
| Git | Latest | https://git-scm.com/ |
| Flutter SDK | Latest stable | https://docs.flutter.dev/get-started/install |
| Raspberry Pi Imager | Latest | https://www.raspberrypi.com/software/ |

Verify your installations:

```bash
docker --version          # Docker version 24+
docker compose version    # Docker Compose version v2+
python --version          # Python 3.12+
node --version            # v22+
npm --version             # 10+
git --version
flutter --version
```

---

## Part 1: Development Machine Setup

### 1.1 Clone and Explore the Repo

```bash
git clone <your-repo-url> exam-conductor
cd exam-conductor
```

Familiarize yourself with the directory structure:

```
exam-conductor/
├── services/           # 16 backend microservices (Python/FastAPI)
│   ├── svc-auth/
│   ├── svc-exam-orch/
│   ├── svc-stroke-ingest/
│   ├── svc-stroke-proc/
│   ├── svc-doc-assembly/
│   ├── svc-ai-pipeline/
│   ├── svc-score-engine/
│   ├── svc-review/
│   ├── svc-analytics/
│   ├── svc-plagiarism/
│   ├── svc-chat/
│   ├── svc-notify/
│   ├── svc-copy-upload/
│   ├── svc-teacher-bff/
│   ├── svc-student-bff/
│   └── svc-invig-console/
├── hub/                # 9 RPi hub modules (Python)
│   ├── hub-common/
│   ├── hub-supervisor/
│   ├── hub-ble-mgr/
│   ├── hub-pen-sync/
│   ├── hub-timer/
│   ├── hub-store/
│   ├── hub-uplink/
│   ├── hub-invig-ble/
│   └── hub-tui/
├── frontend/           # 3 React/TypeScript frontends
│   ├── teacher-dashboard/
│   ├── student-portal/
│   └── invigilator-console/
├── libs/               # Shared libraries
│   ├── exampen-proto/
│   ├── exampen-common-py/
│   └── exampen-common-ts/
├── mobile/             # Flutter mobile app
│   └── exampen-mobile/
├── infra/              # Docker Compose, Traefik, monitoring
├── test-suite/         # Integration, E2E, hub hardware tests
│   └── stoody-mock/    # Mock Stoody server for development
└── scripts/            # Dev setup, seed data, mock generation
```

### 1.2 Install Python Dependencies for Backend Development

Each Python service and library needs its own virtual environment. Never install packages globally.

#### Install shared libraries first (other services depend on these):

```bash
# exampen-proto (protobuf/JSON schema definitions)
cd libs/exampen-proto
python -m venv .venv
.venv/Scripts/activate    # Windows
# source .venv/bin/activate  # Mac/Linux
pip install -e .
deactivate

# exampen-common-py (shared Python utilities: auth, nats, db, logging)
cd ../exampen-common-py
python -m venv .venv
.venv/Scripts/activate    # Windows
# source .venv/bin/activate  # Mac/Linux
pip install -e .
deactivate
```

#### Install a backend service (repeat for each service you work on):

```bash
cd ../../services/svc-auth
python -m venv .venv
.venv/Scripts/activate    # Windows
# source .venv/bin/activate  # Mac/Linux
pip install -e ".[dev]"

# If the service depends on shared libs, install them too:
pip install -e ../../libs/exampen-proto
pip install -e ../../libs/exampen-common-py
deactivate
```

Repeat the same pattern for any service in `services/`. The key services to set up first:

1. `svc-auth` -- JWT validation, role mapping (requires Stoody mock)
2. `svc-exam-orch` -- exam lifecycle FSM
3. `svc-stroke-ingest` -- chunk upload ingestion

#### Install hub packages (for hub development on your PC):

```bash
cd hub/hub-common
python -m venv .venv
.venv/Scripts/activate
pip install -e .
deactivate

# Then for each hub module (hub-store, hub-timer, etc.):
cd ../hub-store
python -m venv .venv
.venv/Scripts/activate
pip install -e .
pip install -e ../hub-common
deactivate
```

### 1.3 Install Node.js Dependencies for Frontend Development

#### Build the shared TypeScript library first:

```bash
cd libs/exampen-common-ts
npm install
npm run build
```

#### Install each frontend:

```bash
# Teacher dashboard
cd ../../frontend/teacher-dashboard
npm install

# Student portal
cd ../student-portal
npm install

# Invigilator console (real-time WebSocket dashboard)
cd ../invigilator-console
npm install
```

### 1.4 Start the Docker Compose Dev Stack

This starts all infrastructure: PostgreSQL + TimescaleDB, NATS JetStream, MinIO (S3), Redis, and Traefik.

```bash
cd infra

# Copy the environment template
cp .env.example .env
```

Review `infra/.env` -- the defaults are fine for local development:

```
POSTGRES_USER=exampen
POSTGRES_PASSWORD=exampen_dev
REDIS_PASSWORD=exampen_redis_dev
MINIO_ROOT_USER=exampen
MINIO_ROOT_PASSWORD=exampen_minio_dev
```

Start the stack:

```bash
docker compose up -d
```

Wait about 30 seconds, then verify all services are healthy:

```bash
docker compose ps
```

Expected output -- every service should show `healthy`:

```
NAME                STATUS              PORTS
exampen-minio-1     Up (healthy)       0.0.0.0:9000->9000/tcp, 0.0.0.0:9001->9001/tcp
exampen-nats-1      Up (healthy)       0.0.0.0:4222->4222/tcp, 0.0.0.0:8222->8222/tcp
exampen-postgres-1  Up (healthy)       0.0.0.0:5432->5432/tcp
exampen-redis-1     Up (healthy)       0.0.0.0:6379->6379/tcp
exampen-traefik-1   Up (healthy)       0.0.0.0:80->80/tcp, 0.0.0.0:8080->8080/tcp
```

The `init-db.sql` script runs automatically on first start and creates 10 service databases:

- `exampen_auth`, `exampen_exam`, `exampen_stroke` (with TimescaleDB), `exampen_score`,
  `exampen_review`, `exampen_analytics`, `exampen_plagiarism`, `exampen_chat`,
  `exampen_copy`, `exampen_notify`

MinIO buckets `exampen-pages` and `exampen-copies` are auto-created by the `minio-init` sidecar.

Verify database creation:

```bash
docker exec exampen-postgres-1 psql -U exampen -d exampen_auth -c "\l" | grep exampen
```

You should see all 10 databases listed.

#### Quick reference -- infrastructure ports:

| Service | Port | URL |
|---------|------|-----|
| PostgreSQL | 5432 | `postgresql://exampen:exampen_dev@localhost:5432/exampen_auth` |
| NATS client | 4222 | `nats://localhost:4222` |
| NATS monitor | 8222 | http://localhost:8222 |
| MinIO API | 9000 | http://localhost:9000 |
| MinIO Console | 9001 | http://localhost:9001 (login: exampen / exampen_minio_dev) |
| Redis | 6379 | `redis://:exampen_redis_dev@localhost:6379/0` |
| Traefik HTTP | 80 | http://localhost |
| Traefik Dashboard | 8080 | http://localhost:8080 |

#### Optional -- add monitoring:

```bash
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

This adds Prometheus (`:9090`), Grafana (`:3000`, login: admin/admin), Loki (`:3100`), and Tempo (`:3200`).

### 1.5 Start the Stoody Mock Server

ExamPen authenticates users via Stoody JWTs. For development, a mock server simulates Stoody's APIs.

```bash
cd test-suite/stoody-mock
pip install fastapi uvicorn pyjwt[crypto] cryptography
uvicorn main:app --port 9100 --reload
```

Verify the JWKS endpoint works:

```bash
curl http://localhost:9100/.well-known/jwks.json
```

Expected output: a JSON object with a `keys` array containing an RSA public key.

Generate a test JWT for development:

```bash
curl -X POST "http://localhost:9100/debug/token?user_id=tutor-001&role=tutor"
```

This returns a signed JWT you can use as a Bearer token for ExamPen API calls.

Available test users:

| user_id | role | name |
|---------|------|------|
| `tutor-001` | tutor | Rajesh Kumar |
| `tutor-002` | tutor | Priya Sharma |
| `student-001` | student | Arjun Mehta |
| `student-002` | student | Sneha Patel |
| `student-003` | student | Rohit Gupta |
| `parent-001` | parent | Vikram Mehta |
| `admin-001` | admin | Dr. Sunita Reddy |

### 1.6 Start Backend Services (Dev Mode)

Each service runs independently with `uvicorn`. Start them in separate terminals.

#### Service port map:

| Service | Port | Purpose |
|---------|------|---------|
| svc-auth | 8000 | JWT validation, role mapping |
| svc-exam-orch | 8001 | Exam lifecycle FSM |
| svc-stroke-ingest | 8002 | Chunk upload ingestion |
| svc-stroke-proc | 8003 | Dedup, normalize, TimescaleDB |
| svc-doc-assembly | 8004 | Stroke-to-page rendering |
| svc-ai-pipeline | 8005 | HWR, step detection, diagram classification |
| svc-score-engine | 8006 | Event-sourced scoring |
| svc-review | 8007 | Objection lifecycle |
| svc-analytics | 8008 | Percentiles, leaderboards |
| svc-plagiarism | 8009 | TF-IDF similarity detection |
| svc-chat | 8010 | Append-only messaging |
| svc-notify | 8011 | Email, push, SMS triggers |
| svc-copy-upload | 8012 | Fallback photo-based capture |
| svc-teacher-bff | 8013 | Read-only aggregator for teacher UI |
| svc-student-bff | 8014 | Read-only aggregator for student UI |
| svc-invig-console | 8015 | WebSocket invigilator dashboard |

To start a service:

```bash
cd services/svc-auth
.venv/Scripts/activate    # Windows
# source .venv/bin/activate  # Mac/Linux

# Set environment variables (point to local Docker infra)
export DATABASE_HOST=localhost
export DATABASE_PORT=5432
export DATABASE_USER=exampen
export DATABASE_PASSWORD=exampen_dev
export NATS_URL=nats://localhost:4222
export MINIO_URL=http://localhost:9000
export REDIS_URL=redis://:exampen_redis_dev@localhost:6379/0

# Start the service
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Repeat for each service you need running, adjusting the port number.

Verify a service is running:

```bash
curl http://localhost:8000/health
```

### 1.7 Start Frontend Dev Servers

Each frontend runs its own Vite dev server.

```bash
# Terminal 1: Teacher dashboard
cd frontend/teacher-dashboard
npm run dev
# Runs on http://localhost:5173

# Terminal 2: Student portal
cd frontend/student-portal
npm run dev
# Runs on http://localhost:5174

# Terminal 3: Invigilator console
cd frontend/invigilator-console
npm run dev
# Runs on http://localhost:5175
```

Vite is configured with proxy rules to forward API calls to the backend services. Check each project's `vite.config.ts` for the exact proxy mappings.

### 1.8 Run the Test Suite

#### Python unit tests (per service):

```bash
cd services/svc-score-engine
.venv/Scripts/activate
pytest tests/ -m unit -v
```

#### Python integration tests (requires Docker Compose stack running):

```bash
# Start the ephemeral test stack (uses separate ports to avoid conflicts)
cd infra
docker compose -f docker-compose.yml -f docker-compose.test.yml up -d

# Run integration tests pointing at the test stack
cd ../services/svc-score-engine
DATABASE_HOST=localhost DATABASE_PORT=5433 NATS_URL=nats://localhost:4223 \
  pytest tests/ -m integration -v

# Tear down the test stack
cd ../../infra
docker compose -f docker-compose.yml -f docker-compose.test.yml down
```

#### TypeScript tests:

```bash
cd frontend/invigilator-console
npm test

cd ../libs/exampen-common-ts
npm test
```

#### Pipeline E2E tests:

```bash
pytest test-suite/pipeline-tests/ -v
```

#### Hub hardware tests (requires RPi):

```bash
pytest test-suite/hub-tests/ -v
```

#### Seed test data:

```bash
./scripts/seed-data.sh --students 40 --exams 3 --questions-per-exam 10
```

---

## Part 2: Raspberry Pi Setup (Headless via P2P Ethernet)

You do NOT need a monitor, keyboard, or mouse for this. Everything is done over SSH via a direct Ethernet cable from your PC to the RPi.

### 2.1 Flash Ubuntu Server 24.04 LTS to SD Card

Start with stock Ubuntu, not the golden image. The golden image is for production deployment after you have verified everything works.

1. Download **Ubuntu Server 24.04 LTS arm64** for Raspberry Pi:
   https://cdimage.ubuntu.com/releases/24.04/release/

   Look for: `ubuntu-24.04-preinstalled-server-arm64+raspi.img.xz`

2. Open **Raspberry Pi Imager** on your PC.

3. Click "Choose OS" --> scroll down --> "Use custom" --> select the downloaded `.img.xz` file.

4. Click "Choose Storage" --> select your micro SD card.

5. Click the gear icon (Advanced options):
   - Set hostname: `exampen-hub`
   - Enable SSH: "Use password authentication"
   - Set username: `ubuntu`
   - Set password: choose something you will remember (e.g., `exampen123`)
   - Set locale: `en_US`
   - Set timezone: `UTC`
   - Skip WiFi configuration for now (we will use Ethernet)

6. Click "Write" and wait for it to finish.

7. Insert the SD card into the RPi. Do NOT power it on yet.

### 2.2 Enable SSH on First Boot

For Ubuntu Server 24.04, SSH is enabled by default when you set the password in Raspberry Pi Imager (step 2.1 above). The imager writes a `cloud-init` user-data file to the boot partition.

If you used balenaEtcher instead of the Imager and did NOT configure cloud-init:

1. Remove the SD card from the RPi (if inserted) and plug it back into your PC.
2. Open the `system-boot` (FAT32) partition on the SD card.
3. Edit the file `user-data` and ensure it contains:

```yaml
#cloud-config
hostname: exampen-hub
manage_etc_hosts: true
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    lock_passwd: false
    # Replace <hash> with output of: openssl passwd -6 exampen123
    passwd: <hash>
ssh_pwauth: true
```

4. Save and safely eject the SD card.

### 2.3 Configure P2P Ethernet (Headless, No Monitor Needed)

This is the most important section. You will connect the RPi directly to your PC with an Ethernet cable and SSH into it.

#### On Windows (primary instructions):

**Step 1: Physical connection**

1. Insert the SD card into the RPi.
2. Plug one end of the Ethernet cable into the RPi's Ethernet port.
3. Plug the other end into your PC's Ethernet port (or a USB-to-Ethernet adapter).
4. Plug in the USB-C power supply to the RPi. The red LED turns on. The green LED blinks as it boots.

**Step 2: Configure your PC's Ethernet adapter (Method A -- Internet Connection Sharing, recommended)**

This method gives the RPi internet access through your PC's WiFi, which you will need for apt updates.

1. Press `Win+R`, type `ncpa.cpl`, press Enter. This opens Network Connections.
2. Find your **WiFi** adapter (the one connected to the internet). Right-click it --> Properties.
3. Click the **Sharing** tab.
4. Check "Allow other network users to connect through this computer's Internet connection."
5. In the dropdown, select your **Ethernet** adapter (the one connected to the RPi).
6. Click OK.

Windows automatically assigns `192.168.137.1` to your Ethernet adapter and runs a DHCP server. The RPi will get an IP in the `192.168.137.x` range.

**Step 3: Wait for the RPi to boot**

First boot takes 2-3 minutes (cloud-init runs, resizes filesystem, generates SSH keys). Wait until the green LED stops blinking rapidly.

**Step 4: Find the RPi's IP address**

Open PowerShell or Command Prompt:

```powershell
arp -a
```

Look for an entry on the `192.168.137.x` interface. The RPi's IP will be something like `192.168.137.100` or `192.168.137.50`.

Example output:
```
Interface: 192.168.137.1 --- 0x5
  Internet Address      Physical Address      Type
  192.168.137.100       dc-a6-32-xx-xx-xx     dynamic   <-- This is the RPi
  192.168.137.255       ff-ff-ff-ff-ff-ff     static
```

If `arp -a` shows nothing, wait another minute and try again. The RPi needs time to complete first boot.

Alternative: try mDNS (works if Avahi/Bonjour is installed):
```powershell
ping exampen-hub.local
```

**Step 5: SSH into the RPi**

```powershell
ssh ubuntu@192.168.137.100
```

Accept the host key fingerprint (type `yes`). Enter the password you set in step 2.1.

If you did NOT set a password via Raspberry Pi Imager, the default Ubuntu password is `ubuntu` and you will be forced to change it immediately on first login.

**(Method B -- Static IP, no internet sharing)**

If you do not want to share internet, configure a static IP instead:

1. Press `Win+R`, type `ncpa.cpl`, press Enter.
2. Right-click your Ethernet adapter --> Properties.
3. Select "Internet Protocol Version 4 (TCP/IPv4)" --> Properties.
4. Select "Use the following IP address":
   - IP address: `192.168.137.1`
   - Subnet mask: `255.255.255.0`
   - Leave gateway and DNS blank.
5. Click OK.

The RPi will not get an IP via DHCP in this case. You will need to configure a static IP on the RPi side via cloud-init (edit `user-data` on the SD card before first boot) or connect with a link-local address. Method A is strongly preferred.

#### On Mac:

1. Connect the Ethernet cable (use a USB-C to Ethernet adapter if needed).
2. System Settings --> Network --> Ethernet --> Configure IPv4: Manually.
   - IP: `192.168.137.1`, Subnet: `255.255.255.0`, Router: leave blank.
3. System Settings --> General --> Sharing --> Internet Sharing:
   - Share from: Wi-Fi
   - To: Ethernet (or USB Ethernet)
   - Turn on Internet Sharing.
4. Wait 2-3 minutes for RPi to boot.
5. Find RPi IP: `arp -a | grep 192.168.137`
6. SSH in: `ssh ubuntu@192.168.137.x`

#### On Linux:

1. Connect the Ethernet cable.
2. Find your Ethernet interface name:
   ```bash
   ip link show
   ```
   It will be something like `eth0`, `enp0s3`, or `enx...`.

3. Assign a static IP:
   ```bash
   sudo ip addr add 192.168.137.1/24 dev eth0
   sudo ip link set eth0 up
   ```

4. Enable internet sharing (NAT):
   ```bash
   sudo sysctl -w net.ipv4.ip_forward=1
   sudo iptables -t nat -A POSTROUTING -o wlan0 -j MASQUERADE
   sudo iptables -A FORWARD -i eth0 -o wlan0 -j ACCEPT
   sudo iptables -A FORWARD -i wlan0 -o eth0 -m state --state RELATED,ESTABLISHED -j ACCEPT
   ```
   Replace `wlan0` with your WiFi interface and `eth0` with your Ethernet interface.

5. Run a DHCP server for the RPi:
   ```bash
   sudo apt install dnsmasq
   sudo tee /tmp/dnsmasq-rpi.conf <<EOF
   interface=eth0
   dhcp-range=192.168.137.100,192.168.137.200,12h
   EOF
   sudo dnsmasq -C /tmp/dnsmasq-rpi.conf --no-daemon &
   ```

6. Wait 2-3 minutes, then SSH: `ssh ubuntu@192.168.137.100`

### 2.4 First-Login RPi Configuration

After SSHing in:

```bash
# Change the default password (if prompted, follow the prompts)
# If not prompted, change it manually:
passwd

# Set hostname
sudo hostnamectl set-hostname exampen-hub-001

# Set timezone to UTC
sudo timedatectl set-timezone UTC

# Verify
timedatectl
# Should show: Time zone: UTC (UTC, +0000)

# Set locale
sudo localectl set-locale LANG=en_US.UTF-8

# Update all packages (this may take 5-10 minutes)
sudo apt update && sudo apt upgrade -y
```

### 2.5 Configure WiFi Regulatory Domain

The WiFi regulatory domain MUST be set to US. This is a locked configuration for ExamPen.

```bash
sudo apt install -y crda wireless-regdb

# Set regulatory domain
echo 'REGDOMAIN=US' | sudo tee /etc/default/crda

# Persist across reboots via kernel module option
echo 'options cfg80211 ieee80211_regdom=US' | sudo tee /etc/modprobe.d/cfg80211.conf

# Apply immediately
sudo iw reg set US

# Verify
iw reg get
```

Expected output should include `country US: DFS-FCC` with channel listings.

### 2.6 Install Required Packages

```bash
sudo apt install -y \
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
    openssh-server
```

Verify BlueZ version (must be 5.72+):

```bash
bluetoothctl --version
```

Expected output: `bluetoothctl: 5.72` or higher.

Enable required services:

```bash
sudo systemctl enable bluetooth
sudo systemctl enable chrony
sudo systemctl enable NetworkManager
sudo systemctl start bluetooth
sudo systemctl start chrony
sudo systemctl start NetworkManager
```

### 2.7 Set Up the Data Partition and USB Mount

#### Format the USB thumb drive:

Plug the USB thumb drive into the RPi (via the powered USB hub or directly).

Find the device name:

```bash
lsblk
```

Look for a device like `/dev/sda` or `/dev/sda1`. Be VERY careful to identify the correct device -- you do not want to format the SD card.

```bash
# Format the USB drive as ext4 with the required label
sudo mkfs.ext4 -L exampen-backup /dev/sda1
```

#### Create required directories:

```bash
sudo mkdir -p /etc/exampen
sudo mkdir -p /var/lib/exampen/data
sudo mkdir -p /var/lib/exampen/logs
sudo mkdir -p /var/log/exampen
sudo mkdir -p /mnt/exampen-backup
sudo mkdir -p /opt/exampen/bin
```

#### Add mount entries to fstab:

```bash
# Append the ExamPen mount entries
sudo tee -a /etc/fstab <<'EOF'

# ExamPen data partition (if using golden image partition layout)
# LABEL=exampen-data  /var/lib/exampen  ext4  defaults,noatime  0  2

# USB backup drive
LABEL=exampen-backup  /mnt/exampen-backup  ext4  defaults,nofail,noatime,nosuid,nodev,noexec  0  2
EOF
```

Note: The `nofail` flag is critical -- it ensures the RPi boots even if the USB drive is missing.

Mount the USB drive now:

```bash
sudo mount -a

# Verify
df -h | grep exampen
```

Expected output shows `/mnt/exampen-backup` mounted.

### 2.8 Install ExamPen Hub Software

#### Create the Python virtual environment:

```bash
sudo mkdir -p /opt/exampen
sudo python3.12 -m venv /opt/exampen/venv
sudo /opt/exampen/venv/bin/pip install --upgrade pip setuptools wheel
```

#### Install hub packages from the repo:

Copy the hub source code to the RPi. From your development PC:

```bash
# From your dev machine, SCP the hub code to the RPi
scp -r hub/ ubuntu@192.168.137.100:/tmp/hub/
scp -r libs/exampen-proto/ ubuntu@192.168.137.100:/tmp/exampen-proto/
scp -r libs/exampen-common-py/ ubuntu@192.168.137.100:/tmp/exampen-common-py/
```

Back on the RPi:

```bash
# Install shared libs first
sudo /opt/exampen/venv/bin/pip install /tmp/exampen-proto/
sudo /opt/exampen/venv/bin/pip install /tmp/exampen-common-py/

# Install hub packages in dependency order
sudo /opt/exampen/venv/bin/pip install /tmp/hub/hub-common/
sudo /opt/exampen/venv/bin/pip install /tmp/hub/hub-store/
sudo /opt/exampen/venv/bin/pip install /tmp/hub/hub-timer/
sudo /opt/exampen/venv/bin/pip install /tmp/hub/hub-ble-mgr/
sudo /opt/exampen/venv/bin/pip install /tmp/hub/hub-pen-sync/
sudo /opt/exampen/venv/bin/pip install /tmp/hub/hub-uplink/
sudo /opt/exampen/venv/bin/pip install /tmp/hub/hub-invig-ble/
sudo /opt/exampen/venv/bin/pip install /tmp/hub/hub-tui/
sudo /opt/exampen/venv/bin/pip install /tmp/hub/hub-supervisor/
```

Or use the installer script (equivalent to the above):

```bash
sudo /tmp/hub/../infra/hub-image/install-hub-software.sh --from-source /tmp/hub
```

#### Create the hub-supervisor wrapper script:

```bash
sudo tee /opt/exampen/bin/hub-supervisor <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
VENV_DIR="/opt/exampen/venv"
export PATH="${VENV_DIR}/bin:${PATH}"
export VIRTUAL_ENV="${VENV_DIR}"
export EXAMPEN_DATA="${EXAMPEN_DATA:-/var/lib/exampen}"
export EXAMPEN_BACKUP="${EXAMPEN_BACKUP:-/mnt/exampen-backup}"
export EXAMPEN_CONFIG="${EXAMPEN_CONFIG:-/etc/exampen/hub.conf}"
export EXAMPEN_LOG_LEVEL="${EXAMPEN_LOG_LEVEL:-INFO}"
exec "${VENV_DIR}/bin/python" -m hub_supervisor "$@"
EOF

sudo chmod +x /opt/exampen/bin/hub-supervisor
```

#### Install the systemd service:

```bash
sudo tee /etc/systemd/system/exampen-supervisor.service <<'EOF'
[Unit]
Description=ExamPen Hub Supervisor
After=network-online.target bluetooth.target
Wants=network-online.target bluetooth.target

[Service]
Type=notify
ExecStart=/opt/exampen/bin/hub-supervisor
Restart=always
RestartSec=5
WatchdogSec=30
StandardOutput=journal
StandardError=journal
Environment=EXAMPEN_DATA=/var/lib/exampen
Environment=EXAMPEN_BACKUP=/mnt/exampen-backup

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable exampen-supervisor.service
```

Do NOT start the service yet -- it needs configuration first (Part 3).

### 2.9 Configure WiFi for School Network (or Keep Using P2P Ethernet)

**For development, keep using the P2P Ethernet connection.** You do not need WiFi to test the hub.

If you do want to connect to WiFi:

```bash
# Scan for networks
nmcli device wifi list

# Connect to a network
sudo nmcli device wifi connect "YourSchoolWiFi" password "your-password"

# Force 5 GHz band preference (less BLE interference)
sudo nmcli connection modify "YourSchoolWiFi" 802-11-wireless.band a

# Verify connection
nmcli device status
ip addr show wlan0
```

If the school network has a captive portal (browser-based login), the hub cannot handle it. Either whitelist the hub's MAC address on the network, or use a mobile hotspot.

### 2.10 Verify Hub Is Running

After completing Part 3 (configuration), start the service:

```bash
sudo systemctl start exampen-supervisor.service

# Check status
sudo systemctl status exampen-supervisor.service
```

Expected output: `Active: active (running)`.

View logs:

```bash
journalctl -u exampen-supervisor -f
```

If the TUI is active (HDMI connected or serial console), you will see the ExamPen Hub TUI with 8 screens: Setup, Status, WiFi, Dongles, Exams, Diagnostics, Logs, Shutdown.

Run diagnostics from SSH (if TUI is not accessible):

```bash
# Check BLE adapters
hciconfig -a

# Check USB devices
lsusb

# Check NTP sync
chronyc tracking

# Check disk mounts
df -h
mount | grep exampen
```

---

## Part 3: Connect Hub to Backend

### 3.1 Configure Hub to Point to Development Backend

The hub needs to know where the backend is. For development with P2P Ethernet, the backend runs on your PC.

From the RPi, your PC's IP is `192.168.137.1`.

Create the hub configuration file:

```bash
sudo tee /etc/exampen/hub.conf <<EOF
[hub]
backend_url = http://192.168.137.1:8001
uplink_mode = wifi
hub_code = DEVHUB000001

[network]
# For P2P Ethernet dev, the backend is reachable via the Ethernet link
backend_reachable_via = ethernet
EOF
```

Replace `192.168.137.1:8001` with the actual address and port where `svc-exam-orch` is running on your PC.

### 3.2 First-Boot Provisioning

When the hub supervisor starts and finds `hub.conf` with a `hub_code` but no `hub_id`, it enters provisioning mode.

The provisioning flow:

1. Hub connects to the backend URL.
2. Hub sends: `POST /api/v1/hubs/provision` with `{"hub_code": "DEVHUB000001"}`.
3. Backend responds: `{"hub_id": "EPH-00042", "institute_id": "...", "invig_codes": [...], "pen_inventory": [...]}`.
4. Hub stores the response in its local SQLite database (`/var/lib/exampen/hub.db`).
5. Hub transitions to `PROVISIONED` state.

For development, you need `svc-exam-orch` running and its provisioning endpoint implemented. If the endpoint is not yet built, you can manually seed the hub database:

```bash
sudo sqlite3 /var/lib/exampen/hub.db <<'SQL'
CREATE TABLE IF NOT EXISTS hub_config (
    hub_id TEXT PRIMARY KEY,
    backend_url TEXT NOT NULL,
    uplink_mode TEXT NOT NULL DEFAULT 'wifi',
    region TEXT NOT NULL DEFAULT 'US',
    provisioned_at TEXT NOT NULL,
    last_backend_sync TEXT
);

INSERT OR REPLACE INTO hub_config VALUES (
    'EPH-DEV-001',
    'http://192.168.137.1:8001',
    'wifi',
    'US',
    datetime('now'),
    NULL
);
SQL
```

### 3.3 Verify Connectivity

Test that the RPi can reach the backend:

```bash
# Test connectivity to your PC
ping -c 3 192.168.137.1

# Test the backend health endpoint (adjust port as needed)
curl -s http://192.168.137.1:8001/health
```

If using the TUI, the Status screen ([2]) shows:
- Backend: Reachable / Unreachable
- WiFi/Ethernet: Connected / Disconnected
- NTP: Synced / Not synced

---

## Part 4: BLE Dongle and Pen Setup

### 4.1 Connect USB BLE Dongles

1. Plug 5 BLE dongles into the powered USB hub.
2. Plug the powered USB hub into the RPi.
3. Make sure the USB hub's external power supply is connected.

Verify the dongles are detected:

```bash
hciconfig -a
```

Expected output (5 adapters):

```
hci0:   Type: Primary  Bus: USB
        BD Address: AA:BB:CC:DD:EE:01  ACL MTU: 310:10
        UP RUNNING

hci1:   Type: Primary  Bus: USB
        BD Address: AA:BB:CC:DD:EE:02  ACL MTU: 310:10
        UP RUNNING

hci2:   ...
hci3:   ...
hci4:   ...
```

If you see fewer than 5, check:
- Is the USB hub powered? (some hubs cannot power 5 dongles without external power)
- Try different USB ports on the hub
- Try `sudo hciconfig hci0 up` for any adapters that show `DOWN`

Also verify via `lsusb`:

```bash
lsusb
```

You should see 5 Bluetooth entries (usually showing as "Cambridge Silicon Radio" or "Realtek").

Bring all adapters up:

```bash
for i in 0 1 2 3 4; do
    sudo hciconfig hci$i up
    sudo hciconfig hci$i piscan   # Make discoverable for pen scan
done
```

### 4.2 Pen Pairing and Registration

The pen registration flow happens during exam setup via the invigilator's mobile app:

1. Power on P05 pens near the hub (within BLE range, ~5 meters).
2. The invigilator opens the ExamPen mobile app --> Hub Control --> Register Pens.
3. The app sends a BLE command to the hub to start a registration scan.
4. The hub's `hub-ble-mgr` scans on all 5 dongles for pens advertising the P05 GATT service (`0000ae30-...`).
5. Discovered pens appear in the TUI Dongle screen and the mobile app.
6. Each dongle can connect up to 8 pens (5 dongles x 8 = 40 pens max per hub).
7. The invigilator assigns students to pens (MAC-to-student mapping).

For development testing, verify BLE scanning works:

```bash
sudo hcitool -i hci0 lescan --passive
```

This lists nearby BLE devices. Press `Ctrl+C` to stop. If P05 pens are nearby and powered on, you will see their MAC addresses.

### 4.3 Pen Simulator (If No Physical Pens)

If you do not have physical P05 pens, you can simulate them.

**Option A: nRF52840-DK (hardware simulator)**

1. Flash the pen simulator firmware onto the nRF52840-DK.
2. The DK advertises itself as a P05 pen with the correct GATT service UUID.
3. The hub's `hub-ble-mgr` sees it as a real pen.
4. Firmware files are in `test-suite/fixtures/` (if available).

**Option B: Software BLE simulator (requires a second BLE adapter)**

If your development PC has a BLE adapter (or you use one of the USB dongles on your PC), you can run a pen simulator script:

```bash
cd test-suite/hub-tests
python ble_pen_sim.py --adapter hci0 --pen-count 5
```

This advertises virtual pens that the hub can discover and sync from.

**Option C: Mock mode**

For testing the hub software without any BLE hardware:

```bash
EXAMPEN_MOCK_BLE=true sudo /opt/exampen/bin/hub-supervisor
```

This starts the hub with simulated BLE responses, useful for testing the supervisor FSM, TUI, and upload pipeline.

---

## Part 5: Running an Exam (End-to-End)

This section walks through a complete exam flow from creation to score publication.

### 5.1 Create Exam via Teacher Dashboard or API

**Via the teacher dashboard:**

1. Open http://localhost:5173 in your browser.
2. Log in using a test JWT (generate one from the Stoody mock server).
3. Navigate to ExamPen --> Create Exam.
4. Fill in: subject, class/section, date, duration, question count, total marks.

**Via API (using curl):**

```bash
# Get a test JWT
TOKEN=$(curl -s -X POST "http://localhost:9100/debug/token?user_id=tutor-001&role=tutor" | jq -r '.token')

# Create an exam
curl -X POST http://localhost:8001/api/v1/exams \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id": "MATH-10",
    "class_id": "10A",
    "date": "2026-03-25",
    "start_time": "09:00",
    "duration_min": 60,
    "question_count": 10,
    "total_marks": 100
  }'
```

### 5.2 Assign Invigilators and Evaluators

```bash
# Assign an invigilator
curl -X POST http://localhost:8001/api/v1/exams/{exam_id}/invigilators \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "tutor-002"}'

# Assign an evaluator
curl -X POST http://localhost:8001/api/v1/exams/{exam_id}/evaluators \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "tutor-001"}'
```

### 5.3 Define Rubric and Question Regions

Via the teacher dashboard:
1. Go to the exam --> Rubric tab.
2. For each question, set: marks allocation, step breakdown, expected answer type.
3. Go to the Question Regions tab --> upload answer sheet layout --> draw bounding boxes for each question.

### 5.4 Invigilator Connects to Hub via Mobile App

1. The invigilator opens the ExamPen mobile app on their phone.
2. Taps "Hub Control" --> scans for nearby hubs via BLE.
3. Selects the hub (e.g., `EPH-DEV-001`).
4. Enters the invigilator code (generated when assigned in step 5.2).
5. The hub authenticates the code against its cached `invig_codes` table.

### 5.5 Arm and Start the Exam

1. Invigilator taps "Register Pens" --> hub scans for BLE pens --> students place pens near the hub.
2. Discovered pens appear in the app. Invigilator assigns each pen to a student.
3. Invigilator taps "Arm Exam" --> exam state changes to `armed`.
4. Invigilator taps "Start Timer" --> exam state changes to `timer_running`.
5. The countdown appears on the TUI Status screen and the mobile app.

### 5.6 Students Write with Pens

Students write on dot-matrix answer sheets using P05 pens. The pens store all stroke data locally in flash memory. No BLE connection needed during the exam.

The hub's timer runs locally using `CLOCK_MONOTONIC`. It persists to SQLite every 10 seconds, so it survives reboots. WiFi is not required during the exam.

### 5.7 Timer Expires --> Dongle Activation --> Pen Sync

1. Timer reaches zero --> exam state changes to `dongle_activation`.
2. Hub activates all 5 BLE dongles to scan for pens.
3. Each dongle connects to up to 8 pens simultaneously.
4. Hub reads stroke data from each pen via GATT (`hub-pen-sync`).
5. Data is dual-written: SD card (`/var/lib/exampen/data/`) first, then USB drive (`/mnt/exampen-backup/data/`). Both writes must `fsync()` before ACKing the pen.
6. Progress appears on TUI Status screen and mobile app.
7. Once all pens are synced (or timeout reached), exam state changes to `sync_complete` (or `sync_partial` if some pens failed).

### 5.8 Hub Uploads Data to Backend

1. Hub checks WiFi/Ethernet connectivity.
2. `hub-uplink` reads the upload ledger and sends data to `svc-stroke-ingest` in chunks.
3. Each chunk is ACKd by the backend. The ledger tracks which chunks have been ACKd (resumable).
4. Once all chunks are uploaded, exam state changes to `upload_complete`.
5. If WiFi is unavailable, the mobile app can relay data via BLE (slower fallback path).

### 5.9 AI Processing --> Scoring

Once strokes are ingested, the backend pipeline runs automatically via NATS events:

1. `svc-stroke-ingest` publishes `stroke.raw` events.
2. `svc-stroke-proc` deduplicates, normalizes, and commits to TimescaleDB --> publishes `stroke.processed`.
3. `svc-doc-assembly` renders strokes into page images --> publishes `page.ready`.
4. `svc-ai-pipeline` runs HWR (handwriting recognition), step detection, and diagram classification --> publishes `ai.result`.
5. `svc-score-engine` applies rubric rules to AI output, generates scores --> publishes `score.updated`.

### 5.10 Teacher Reviews Scores

1. Teacher opens the dashboard --> Exam --> Score Review.
2. Sees class overview: all students, AI scores, confidence percentages, miss indicators, plagiarism flags.
3. Drills into individual students to review per-question breakdowns.
4. Edits scores where AI was wrong (mandatory reason required for each edit).
5. Uses "Bulk Approve" for high-confidence answers above threshold.
6. Reviews plagiarism flags (side-by-side comparison, confirm or dismiss).
7. When satisfied, clicks "Finalize Scores" --> scores are locked.

### 5.11 Publish Scores --> Students View Results

1. Teacher clicks "Publish" --> scores transition to `finalized` --> `locked`.
2. `svc-notify` sends notifications to students (email, push).
3. ExamPen pushes score summary to Stoody via webhook: `POST /api/webhooks/exampen/scores`.
4. Students log in to the student portal --> ExamPen tab --> see their scores.
5. Students can view: total score, question-wise breakdown, answer images, AI analysis.
6. If the objection window is open, students can file objections on specific questions.

---

## Part 6: Troubleshooting

### Hub Won't Boot

| Symptom | Check | Fix |
|---------|-------|-----|
| No LEDs at all | Power supply | Use a 5V 3A (RPi 4B) or 5V 5A (RPi 5) USB-C supply. Some phone chargers are too weak. |
| Red LED only, no green blinking | SD card | Re-flash the SD card. Try a different card. Ensure it is fully seated. |
| Green LED blinks in a pattern (4 blinks) | Kernel not found | The SD card image is corrupt. Re-flash. |
| Boots but shuts down after ~30s | Overheating or power | Check for thermal throttling. Ensure adequate power. |

### Can't SSH to RPi

| Symptom | Check | Fix |
|---------|-------|-----|
| `Connection refused` | SSH service | Re-flash with SSH enabled. Check cloud-init `user-data`. |
| `No route to host` | IP configuration | Verify ICS is enabled on your WiFi adapter. Run `arp -a` to find the RPi. Wait longer (first boot takes 2-3 min). |
| `Connection timed out` | Cable / adapter | Try a different Ethernet cable. Check both ends are firmly plugged in. |
| `Permission denied` | Password | Default is `ubuntu` / `ubuntu`. If you set one in Imager, use that. |
| RPi has no IP in `arp -a` | DHCP not working | Disable and re-enable ICS on your WiFi adapter. Reboot the RPi. |

### BLE Dongles Not Detected

| Symptom | Check | Fix |
|---------|-------|-----|
| `hciconfig` shows no devices | USB hub | Use a POWERED USB hub. Unpowered hubs cannot drive 5 BLE dongles. |
| `hciconfig` shows fewer than 5 | Dongle compatibility | Not all BLE dongles are Linux-compatible. Try CSR8510 or RTL8761B chipsets. |
| `hciconfig hciX: DOWN` | BlueZ service | `sudo systemctl restart bluetooth` then `sudo hciconfig hciX up`. |

### Backend Unreachable from Hub

| Symptom | Check | Fix |
|---------|-------|-----|
| `ping 192.168.137.1` fails | Ethernet link | Check cable. Verify `ip addr` shows an IP on the RPi's `eth0`. |
| Ping works but `curl` fails | Firewall | On Windows: allow the backend port through Windows Firewall. On Linux: check `iptables`. |
| `curl` gets connection refused | Service not running | Ensure the backend service is started on your PC. Check the port number. |
| SSL errors | Wrong protocol | Use `http://` not `https://` for local dev. |

### Pen Not Syncing

| Symptom | Check | Fix |
|---------|-------|-----|
| Pen not discovered | BLE range | Move pen within 5 meters of the hub. Ensure pen is powered on. |
| Pen discovered but won't connect | Dongle capacity | Each dongle supports max 8 simultaneous connections. Check `hub-ble-mgr` logs. |
| Sync starts but fails partway | Battery | Check pen battery level. Low battery causes BLE disconnects. |
| Checksum mismatch | Interference | Move the hub away from WiFi routers (2.4 GHz interference). Use 5 GHz for WiFi. |

### Timer Drift

```bash
# Check NTP sync status
chronyc tracking
```

The `Leap status` should say `Normal`. If it says `Not synchronised`, the RPi cannot reach an NTP server:

```bash
# Check chrony sources
chronyc sources

# Force a sync
sudo chronyc makestep
```

For P2P Ethernet with no internet, chrony will not sync. The timer uses `CLOCK_MONOTONIC` locally, so drift is minimal (seconds per day on RPi hardware).

### Dual-Write Failure

```bash
# Check if USB is mounted
mount | grep exampen-backup

# Check USB drive health
sudo smartctl -a /dev/sda   # (install smartmontools first)

# Check SD card health
sudo dmesg | grep -i "error\|fail\|mmc"
```

If the USB drive fails, `hub-store` degrades to SD-only mode and shows an amber warning on the TUI and invigilator app. Data is still captured, just without the secondary backup.

---

## Part 7: Development Workflow

### Modify a Backend Service and Test

```bash
# 1. Edit code in services/svc-score-engine/src/
# 2. The uvicorn --reload flag auto-restarts on file changes
# 3. Run unit tests:
cd services/svc-score-engine
.venv/Scripts/activate
pytest tests/ -m unit -v

# 4. Run integration tests (Docker stack must be up):
DATABASE_HOST=localhost DATABASE_PORT=5432 \
  pytest tests/ -m integration -v
```

### Modify a Frontend and Test

```bash
# 1. Edit code in frontend/teacher-dashboard/src/
# 2. Vite HMR auto-reloads the browser on save
# 3. Run lint and typecheck:
cd frontend/teacher-dashboard
npm run lint
npm run typecheck

# 4. Build for production:
npm run build
```

### Modify Hub Software and Deploy to RPi

```bash
# 1. Edit code in hub/hub-store/src/ (on your dev PC)

# 2. Re-install the changed package on the RPi:
scp -r hub/hub-store/ ubuntu@192.168.137.100:/tmp/hub-store/
ssh ubuntu@192.168.137.100 "sudo /opt/exampen/venv/bin/pip install /tmp/hub-store/"

# 3. Restart the supervisor to pick up changes:
ssh ubuntu@192.168.137.100 "sudo systemctl restart exampen-supervisor"

# 4. Check logs:
ssh ubuntu@192.168.137.100 "journalctl -u exampen-supervisor -f"
```

For faster iteration, you can also develop hub modules locally on your PC (without BLE hardware) using mock mode.

### Run the Full Test Suite

```bash
# Unit tests for all Python services
for svc in services/svc-*/; do
    echo "Testing $svc..."
    cd "$svc"
    .venv/Scripts/activate
    pytest tests/ -m unit -v
    deactivate
    cd ../..
done

# TypeScript lint + typecheck
for fe in frontend/*/; do
    echo "Checking $fe..."
    cd "$fe"
    npm run lint
    npm run typecheck
    cd ../..
done

# Pipeline E2E tests
pytest test-suite/pipeline-tests/ -v

# Hub hardware tests (requires RPi)
pytest test-suite/hub-tests/ -v
```

### Add a New Test

1. Create the test file in the appropriate location:
   - Unit test: `services/svc-{name}/tests/test_{feature}.py`
   - Integration test: same location, mark with `@pytest.mark.integration`
   - E2E test: `test-suite/pipeline-tests/test_{scenario}.py`
   - Hub test: `test-suite/hub-tests/test_{feature}.py`

2. Use explicit test IDs (e.g., `U-SCR-01`, `I-ORCH-02`, `E2E-08`).

3. Reference test IDs in PRs and docs.

4. State validation levels achieved in your PR:
   - L1: Docker image builds
   - L2: Typecheck/lint pass
   - L3: Unit tests pass (domain logic, no I/O)
   - L4: Integration tests pass (real DB/NATS/S3)
   - L5: E2E tests pass (multi-service pipeline)
   - L6: Hardware-in-loop verified (hub + BLE dongles)
   - L7: Field trial verified (real exam)

### Reset Everything

```bash
# Reset Docker volumes (databases, message queues, object storage)
cd infra
docker compose down -v
docker compose up -d

# Reset RPi hub database
ssh ubuntu@192.168.137.100 "sudo rm /var/lib/exampen/hub.db && sudo systemctl restart exampen-supervisor"

# Reset USB backup data
ssh ubuntu@192.168.137.100 "sudo rm -rf /mnt/exampen-backup/data/* && sudo rm -f /mnt/exampen-backup/hub.db.backup"
```

---

## Quick Reference: Port Map

| Service | Port | Type |
|---------|------|------|
| PostgreSQL | 5432 | TCP |
| NATS | 4222 | TCP |
| NATS Monitor | 8222 | HTTP |
| MinIO API | 9000 | HTTP |
| MinIO Console | 9001 | HTTP |
| Redis | 6379 | TCP |
| Traefik | 80 | HTTP |
| Traefik Dashboard | 8080 | HTTP |
| Stoody Mock | 9100 | HTTP |
| svc-auth | 8000 | HTTP |
| svc-exam-orch | 8001 | HTTP |
| svc-stroke-ingest | 8002 | HTTP |
| svc-stroke-proc | 8003 | HTTP |
| svc-doc-assembly | 8004 | HTTP |
| svc-ai-pipeline | 8005 | HTTP |
| svc-score-engine | 8006 | HTTP |
| svc-review | 8007 | HTTP |
| svc-analytics | 8008 | HTTP |
| svc-plagiarism | 8009 | HTTP |
| svc-chat | 8010 | HTTP |
| svc-notify | 8011 | HTTP |
| svc-copy-upload | 8012 | HTTP |
| svc-teacher-bff | 8013 | HTTP |
| svc-student-bff | 8014 | HTTP |
| svc-invig-console | 8015 | WS |
| Prometheus | 9090 | HTTP |
| Grafana | 3000 | HTTP |
| Loki | 3100 | HTTP |
| Tempo | 3200 | HTTP |
| Teacher Dashboard | 5173 | HTTP |
| Student Portal | 5174 | HTTP |
| Invigilator Console | 5175 | HTTP |

---

## Key File References

| Purpose | File Path |
|---------|-----------|
| Project overview | `CLAUDE.md` |
| Docker dev stack | `infra/docker-compose.yml` |
| Docker prod stack | `infra/docker-compose.prod.yml` |
| Environment template | `infra/.env.example` |
| Database init | `infra/init-db.sql` |
| Hub deployment spec | `new-docs/HUB_DEPLOYMENT_SPEC.md` |
| Golden image builder | `infra/hub-image/build-image.sh` |
| Hub software installer | `infra/hub-image/install-hub-software.sh` |
| Supervisor systemd unit | `infra/hub-image/systemd/exampen-supervisor.service` |
| fstab mount entries | `infra/hub-image/config/fstab.append` |
| Stoody integration spec | `new-docs/STOODY_INTEGRATION_SPEC.md` |
| Stoody mock server | `test-suite/stoody-mock/` |
| Seed data script | `scripts/seed-data.sh` |
| Infra dev README | `infra/README.md` |
| Infra prod README | `infra/README-prod.md` |
