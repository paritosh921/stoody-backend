# P05 Pen SDK - Command Reference

> **Source**: P05_pen_SDK.docx
> **Purpose**: Complete BLE protocol specification for Stoody P05 pen

This is a protocol reference, not an architecture authority. Use it to support the shared ingest substrate, not to infer DCR/PCR evaluation ownership.

---

## BLE GATT Architecture (Verified Against Hardware)

The pen advertises service `0000ae30-...` with these characteristics:

| UUID | Properties | Purpose |
|------|-----------|---------|
| **AE01** | write-without-response | Coordinate data write (pen → app real-time path) |
| **AE02** | notify | **Primary notification channel** — coordinates AND command responses |
| **AE03** | write-without-response | Unused (pen ignores writes here) |
| **AE04** | notify | Secondary notify (pen does not send responses here in practice) |
| **AE05** | indicate | Indication channel |
| **AE10** | read, write | **Command write + device info read** |

There is also a secondary service `0000ae3a-...` with AE3B (write) and AE3C (notify).

### Key findings (confirmed by hardware testing):

- **Commands must be written to AE10** using write-with-response (`response=True`). AE01 and AE03 accept writes but the pen firmware ignores command frames sent there.
- **All responses arrive on AE02** (the coordinate notify channel), NOT AE04. The Cmd byte distinguishes coordinate data (0x00-0x02) from command responses (0x03+).
- **Device info** is also read from AE10 as semicolon-separated key=value pairs: `battery=88;model_name=P05;soft_ver=V2.7.15_260124;mac=B4:CD:BA:E5:0E:C6`
- **Bleak on Windows** requires a sync notification callback wrapper (`loop.call_soon_threadsafe`) instead of native async callbacks due to `WindowsSelectorEventLoopPolicy`.

---

## Traffic Channel Frame Format

| Field | Size | Description |
|-------|------|-------------|
| **Head** | 2B | Frame header: `0x5A, 0x5B` (same for all frames) |
| **SerialNum** | 4B | Frame sequence number (little-endian) |
| **ID** | 4B | Student user identifier (little-endian) |
| **Cmd** | 1B | Command code |
| **DataFormat** | 1B | `0x00` = byte stream, `0x01` = JSON |
| **DataLen** | 2B | Length of Data field (little-endian) |
| **Data** | NB | Payload (variable length) |
| **CRC16** | 2B | CRC-16/XMODEM over SerialNum through Data (little-endian) |

**CRC-16/XMODEM**: polynomial `0x1021`, init `0x0000`, no reflection. Confirmed by brute-force matching against actual pen frames.

**Total Header Size**: 14 bytes (before Data) — Head(2) + SerialNum(4) + ID(4) + Cmd(1) + DataFormat(1) + DataLen(2)

---

## Command List

### Real-time Stroke Commands (0x00-0x02)

| Cmd | Name | Direction | DataFormat | Description |
|-----|------|-----------|------------|-------------|
| `0x00` | Pen Down | Pen→App | Byte | Stroke start, SerialNum=0 |
| `0x01` | Coordinate Data | Pen→App | Byte | 14-byte coordinate packets |
| `0x02` | Pen Up | Pen→App | Byte | Stroke end |

### Device Info Commands (0x03-0x04)

| Cmd | Name | Direction | DataFormat | Description |
|-----|------|-----------|------------|-------------|
| `0x03` | Device Info | App→Pen | JSON | Get version and SN |
| `0x04` | Battery Level | Pen→App | JSON | Battery percentage |

### Offline Data Commands (0x05-0x0B)

| Cmd | Name | Direction | DataFormat | Description |
|-----|------|-----------|------------|-------------|
| `0x05` | Transmission Recording | Pen→App | Byte | Active recording transmission |
| `0x07` | Offline Coordinates | Pen→App | Byte | Offline data packets (14-byte units) |
| `0x08` | Check Offline Size | App→Pen | JSON | Query offline data size |
| `0x09` | Request Offline Data | App→Pen | - | Trigger offline transmission |
| `0x0A` | Delete Offline Data | App→Pen | - | Clear pen memory |
| `0x0B` | Transfer Complete | Pen→App | Byte | Offline transfer finished |

### OTA Update Commands (0x0C-0x0F)

| Cmd | Name | Direction | DataFormat | Description |
|-----|------|-----------|------------|-------------|
| `0x0C` | OTA Upgrade | App→Pen | JSON/Byte | Firmware file transfer |
| `0x0D` | OTA Progress | Pen→App | JSON | Update progress % |
| `0x0E` | OTA Status | Pen→App | JSON | Success/failure report |
| `0x0F` | OTA Response | Pen→App | Byte | Packet acknowledgment |

### Account & Diagnostics (0x10-0x11, 0xE0, 0xE8)

| Cmd | Name | Direction | DataFormat | Description |
|-----|------|-----------|------------|-------------|
| `0x10` | Binding Change | App→Pen | JSON | Update student ID |
| `0x11` | Raw Trajectory | App→Pen | Byte | Get raw trajectory file |
| `0xE0` | Factory Mode | App→Pen | JSON | Enter factory/calibration mode |
| `0xE8` | Recognition Rate | Pen→App | JSON | Per-stroke recognition metrics |

---

## Coordinate Data Frame (14 bytes)

```c
struct DATA_FRAME {
    u8  u8BookType;     // 0x00-0x0E: SS, SN, SM, SL, SW, MS, MN, MM, ML, MW, LS, LN, LM, LL, LW
    u8  u8BookSeq;      // Book serial number (0-127)
    u16 u16PageNo;      // Page number (max 511)
    u16 u16CoordX;      // X coordinate
    u16 u16CoordY;      // Y coordinate
    u8  u8Pressure;     // Pen pressure
    u8  u8PenProp;      // 1=Down, 2=Up
    u32 u32Timestamp;   // 10ms tick count
};
```

---

## Offline Data Sync Flow

```
1. App → Pen: Send 0x08 (check offline size)
2. Pen → App: Respond with {"offline_total_size": 123456}
3. If size > 0:
   App → Pen: Send 0x09 (request offline data)
4. Pen → App: Send 0x07 packets (SerialNum starts at 1)
   - Each packet: App → Pen: ACK with matching SerialNum
   - Timeout: 5 seconds per packet
5. Pen → App: Send 0x0B (transfer complete)
6. App → Pen: Send 0x0A (delete offline data) [optional]
```

---

## OTA Firmware Update Flow

```
1. App: GET https://szxzy.oss-cn-shenzhen.aliyuncs.com/DotMatrixPen/P05/CS/update.json
2. Compare versions, download firmware if newer
3. App → Pen: Send 0x0C with SerialNum=1
   Data: {"file_size": 360448, "file_md5": "..."}
4. Pen → App: Respond {"isReady": true/false, "reason": "..."}
5. If ready:
   App → Pen: Send 0x0C packets (228 bytes each, SerialNum increments)
   Pen → App: Respond 0x0F for each packet
6. App → Pen: Send 0x0C with SerialNum=0xFFFFFFFF (complete)
7. Pen → App: Send 0x0D progress updates
8. Pen → App: Send 0x0E final status {"ota_status": 0}
```

---

## JSON Response Examples

### Device Info (0x03)
```json
{"device_version": "v1.0.1_240605", "sn": "C224060100000001"}
```

### Battery (0x04)
```json
{"device_battery": 90}
```

### Offline Size (0x08)
```json
{"offline_total_size": 123456}
```

### OTA Ready (0x0C response)
```json
{"isReady": true, "reason": ""}
```

### OTA Progress (0x0D)
```json
{"update_progress": 50}
```

### OTA Status (0x0E)
```json
{"ota_status": 0}
```

### Factory Mode (0xE0)
```json
{
  "device_version": "V1.1.0_240811",
  "mac": "aa:bb:cc:dd:ee:ff",
  "sensor_cal": false,
  "dimming_cal": false
}
```
