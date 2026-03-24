/// Data models for pen discovery and binding state.
///
/// The MAC list characteristic (`6f5f2004-...`) sends a JSON array of
/// pen discovery rows.  These models also cover manual-register payloads
/// and binding lifecycle status values from HUB_DEPLOYMENT_SPEC §4.3.
library;

import 'dart:convert';

// ---------------------------------------------------------------------------
// Binding status — mirrors hub SQLite `pen_bindings.status`
// ---------------------------------------------------------------------------

enum PenBindingStatus {
  discovered,
  provisional,
  confirmed,
  rejected;

  static PenBindingStatus fromString(String value) {
    return PenBindingStatus.values.firstWhere(
      (s) => s.name == value,
      orElse: () => PenBindingStatus.discovered,
    );
  }
}

// ---------------------------------------------------------------------------
// Binding source — mirrors hub SQLite `pen_bindings.source`
// ---------------------------------------------------------------------------

enum PenBindingSource {
  scan,
  manualRegister,
  serverSync;

  String get wireValue {
    switch (this) {
      case PenBindingSource.scan:
        return 'scan';
      case PenBindingSource.manualRegister:
        return 'manual_register';
      case PenBindingSource.serverSync:
        return 'server_sync';
    }
  }

  static PenBindingSource fromString(String value) {
    switch (value) {
      case 'manual_register':
        return PenBindingSource.manualRegister;
      case 'server_sync':
        return PenBindingSource.serverSync;
      default:
        return PenBindingSource.scan;
    }
  }
}

// ---------------------------------------------------------------------------
// PenInfo — a single pen as reported by the hub MAC list characteristic
// ---------------------------------------------------------------------------

class PenInfo {
  final String mac;
  final int? rssi;
  final int? batteryPct;
  final String? fwVersion;
  final String? studentId;
  final String? studentName;
  final String? studentRoll;
  final PenBindingStatus status;
  final PenBindingSource source;

  const PenInfo({
    required this.mac,
    this.rssi,
    this.batteryPct,
    this.fwVersion,
    this.studentId,
    this.studentName,
    this.studentRoll,
    this.status = PenBindingStatus.discovered,
    this.source = PenBindingSource.scan,
  });

  factory PenInfo.fromJson(Map<String, dynamic> json) {
    return PenInfo(
      mac: json['pen_mac'] as String? ?? json['mac'] as String? ?? '',
      rssi: json['rssi'] as int?,
      batteryPct: json['battery_pct'] as int?,
      fwVersion: json['fw_version'] as String?,
      studentId: json['student_id'] as String?,
      studentName: json['student_name'] as String?,
      studentRoll: json['student_roll'] as String?,
      status: PenBindingStatus.fromString(
        json['status'] as String? ?? 'discovered',
      ),
      source: PenBindingSource.fromString(
        json['source'] as String? ?? 'scan',
      ),
    );
  }

  Map<String, dynamic> toJson() => {
        'pen_mac': mac,
        if (rssi != null) 'rssi': rssi,
        if (batteryPct != null) 'battery_pct': batteryPct,
        if (fwVersion != null) 'fw_version': fwVersion,
        if (studentId != null) 'student_id': studentId,
        if (studentName != null) 'student_name': studentName,
        if (studentRoll != null) 'student_roll': studentRoll,
        'status': status.name,
        'source': source.wireValue,
      };

  /// Parse the MAC-list characteristic value (UTF-8 JSON array).
  static List<PenInfo> listFromBytes(List<int> bytes) {
    final text = utf8.decode(bytes);
    final list = jsonDecode(text) as List<dynamic>;
    return list
        .cast<Map<String, dynamic>>()
        .map(PenInfo.fromJson)
        .toList(growable: false);
  }

  /// Shortened MAC for display: last 4 hex chars.
  String get shortMac {
    final parts = mac.split(':');
    if (parts.length >= 2) {
      return '${parts[parts.length - 2]}:${parts.last}';
    }
    return mac;
  }

  /// Battery icon hint.
  String get batteryLabel {
    if (batteryPct == null) return '--';
    return '$batteryPct%';
  }
}
