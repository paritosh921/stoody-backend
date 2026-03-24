/// Data models for hub status feed (BLE GATT §5).
///
/// The status feed characteristic (`6f5f2003-...`) sends a 1 Hz UTF-8 JSON
/// object. These classes mirror that schema.
library;

import 'dart:convert';

// ---------------------------------------------------------------------------
// WiFi sub-model
// ---------------------------------------------------------------------------

class WifiStatus {
  final bool connected;
  final String? band;
  final int? signalDbm;

  const WifiStatus({
    required this.connected,
    this.band,
    this.signalDbm,
  });

  factory WifiStatus.fromJson(Map<String, dynamic> json) {
    return WifiStatus(
      connected: json['connected'] as bool? ?? false,
      band: json['band'] as String?,
      signalDbm: json['signal_dbm'] as int?,
    );
  }

  Map<String, dynamic> toJson() => {
        'connected': connected,
        if (band != null) 'band': band,
        if (signalDbm != null) 'signal_dbm': signalDbm,
      };
}

// ---------------------------------------------------------------------------
// Storage sub-model
// ---------------------------------------------------------------------------

class StorageStatus {
  final bool sdOk;
  final bool usbOk;
  final bool degraded;

  const StorageStatus({
    required this.sdOk,
    required this.usbOk,
    required this.degraded,
  });

  factory StorageStatus.fromJson(Map<String, dynamic> json) {
    return StorageStatus(
      sdOk: json['sd_ok'] as bool? ?? true,
      usbOk: json['usb_ok'] as bool? ?? true,
      degraded: json['degraded'] as bool? ?? false,
    );
  }

  Map<String, dynamic> toJson() => {
        'sd_ok': sdOk,
        'usb_ok': usbOk,
        'degraded': degraded,
      };
}

// ---------------------------------------------------------------------------
// Sync summary sub-model
// ---------------------------------------------------------------------------

class SyncSummary {
  final int complete;
  final int inProgress;
  final int failed;
  final int pending;

  const SyncSummary({
    required this.complete,
    required this.inProgress,
    required this.failed,
    required this.pending,
  });

  int get total => complete + inProgress + failed + pending;

  double get progressFraction =>
      total == 0 ? 0.0 : complete / total;

  factory SyncSummary.fromJson(Map<String, dynamic> json) {
    return SyncSummary(
      complete: json['complete'] as int? ?? 0,
      inProgress: json['in_progress'] as int? ?? 0,
      failed: json['failed'] as int? ?? 0,
      pending: json['pending'] as int? ?? 0,
    );
  }

  Map<String, dynamic> toJson() => {
        'complete': complete,
        'in_progress': inProgress,
        'failed': failed,
        'pending': pending,
      };
}

// ---------------------------------------------------------------------------
// Hub status (top-level status feed object)
// ---------------------------------------------------------------------------

class HubStatus {
  final String examId;
  final String state;
  final int timerRemainingSec;
  final WifiStatus wifi;
  final StorageStatus storage;
  final SyncSummary sync;

  const HubStatus({
    required this.examId,
    required this.state,
    required this.timerRemainingSec,
    required this.wifi,
    required this.storage,
    required this.sync,
  });

  factory HubStatus.fromJson(Map<String, dynamic> json) {
    return HubStatus(
      examId: json['exam_id'] as String? ?? '',
      state: json['state'] as String? ?? 'unknown',
      timerRemainingSec: json['timer_remaining_sec'] as int? ?? 0,
      wifi: WifiStatus.fromJson(
        json['wifi'] as Map<String, dynamic>? ?? {},
      ),
      storage: StorageStatus.fromJson(
        json['storage'] as Map<String, dynamic>? ?? {},
      ),
      sync: SyncSummary.fromJson(
        json['sync'] as Map<String, dynamic>? ?? {},
      ),
    );
  }

  factory HubStatus.fromUtf8Bytes(List<int> bytes) {
    final text = utf8.decode(bytes);
    final json = jsonDecode(text) as Map<String, dynamic>;
    return HubStatus.fromJson(json);
  }

  Map<String, dynamic> toJson() => {
        'exam_id': examId,
        'state': state,
        'timer_remaining_sec': timerRemainingSec,
        'wifi': wifi.toJson(),
        'storage': storage.toJson(),
        'sync': sync.toJson(),
      };

  /// Human-readable timer string, e.g. "47:23".
  String get timerDisplay {
    final m = timerRemainingSec ~/ 60;
    final s = timerRemainingSec % 60;
    return '${m.toString().padLeft(2, '0')}:'
        '${s.toString().padLeft(2, '0')}';
  }
}
