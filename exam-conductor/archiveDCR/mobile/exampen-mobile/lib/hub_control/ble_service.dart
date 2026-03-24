/// BLE service for the Invigilator GATT Service (§2 of ble-gatt-spec.md).
///
/// Handles scan, connect, GATT discovery, characteristic read/write,
/// and notification subscriptions for the hub's invigilator service.
///
/// GATT UUIDs (Section 2):
///   Service:     6f5f0002-4d8b-4d8d-9d7d-000000000002
///   Auth:        6f5f2001-4d8b-4d8d-9d7d-000000000002  (Write, Indicate)
///   Command:     6f5f2002-4d8b-4d8d-9d7d-000000000002  (Write)
///   Status feed: 6f5f2003-4d8b-4d8d-9d7d-000000000002  (Notify)
///   MAC list:    6f5f2004-4d8b-4d8d-9d7d-000000000002  (Read, Notify)
///   Data relay:  6f5f2005-4d8b-4d8d-9d7d-000000000002  (Notify)
library;

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_blue_plus/flutter_blue_plus.dart';

import 'models/hub_status.dart';
import 'models/pen_info.dart';

// ---------------------------------------------------------------------------
// GATT UUID constants — Invigilator Service (§2)
// ---------------------------------------------------------------------------

class HubGattUuids {
  HubGattUuids._();

  static final Guid service =
      Guid('6f5f0002-4d8b-4d8d-9d7d-000000000002');
  static final Guid auth =
      Guid('6f5f2001-4d8b-4d8d-9d7d-000000000002');
  static final Guid command =
      Guid('6f5f2002-4d8b-4d8d-9d7d-000000000002');
  static final Guid statusFeed =
      Guid('6f5f2003-4d8b-4d8d-9d7d-000000000002');
  static final Guid macList =
      Guid('6f5f2004-4d8b-4d8d-9d7d-000000000002');
  static final Guid dataRelay =
      Guid('6f5f2005-4d8b-4d8d-9d7d-000000000002');
}

// ---------------------------------------------------------------------------
// Command IDs (§4)
// ---------------------------------------------------------------------------

class HubCommandId {
  HubCommandId._();

  static const int startExam = 0x01;
  static const int stopExam = 0x02;
  static const int startRegistrationScan = 0x03;
  static const int manualRegister = 0x04;
  static const int startUpload = 0x05;
  static const int requestSnapshot = 0x06;
}

// ---------------------------------------------------------------------------
// Auth result from the hub's Indicate response
// ---------------------------------------------------------------------------

enum HubAuthResult {
  accepted,
  rejected,
  timeout;
}

// ---------------------------------------------------------------------------
// BLE Service
// ---------------------------------------------------------------------------

class BleService {
  BluetoothDevice? _device;
  BluetoothCharacteristic? _authChar;
  BluetoothCharacteristic? _commandChar;
  BluetoothCharacteristic? _statusFeedChar;
  BluetoothCharacteristic? _macListChar;

  StreamSubscription<List<int>>? _statusFeedSub;
  StreamSubscription<List<int>>? _macListSub;

  final StreamController<HubStatus> _statusController =
      StreamController<HubStatus>.broadcast();
  final StreamController<List<PenInfo>> _penListController =
      StreamController<List<PenInfo>>.broadcast();

  /// 1 Hz hub status updates.
  Stream<HubStatus> get statusStream => _statusController.stream;

  /// Pen list updates (after scan or registration change).
  Stream<List<PenInfo>> get penListStream => _penListController.stream;

  bool get isConnected => _device?.isConnected ?? false;

  // ---------------------------------------------------------------------------
  // Scan
  // ---------------------------------------------------------------------------

  /// Scan for hubs advertising the invigilator service UUID.
  /// Returns a stream of discovered devices (deduplicated by platform ID).
  Stream<BluetoothDevice> scanForHubs({
    Duration timeout = const Duration(seconds: 10),
  }) async* {
    await FlutterBluePlus.startScan(
      withServices: [HubGattUuids.service],
      timeout: timeout,
    );

    final seen = <String>{};
    await for (final result in FlutterBluePlus.scanResults) {
      for (final r in result) {
        final id = r.device.remoteId.str;
        if (seen.add(id)) {
          yield r.device;
        }
      }
    }
  }

  /// Stop an active scan.
  Future<void> stopScan() => FlutterBluePlus.stopScan();

  // ---------------------------------------------------------------------------
  // Connect & GATT discovery
  // ---------------------------------------------------------------------------

  Future<void> connect(BluetoothDevice device) async {
    await device.connect(
      autoConnect: false,
      timeout: const Duration(seconds: 15),
      mtu: 247,
    );
    _device = device;

    final services = await device.discoverServices();
    final svc = services.firstWhere(
      (s) => s.uuid == HubGattUuids.service,
      orElse: () =>
          throw StateError('Invigilator service not found on device'),
    );

    for (final c in svc.characteristics) {
      if (c.uuid == HubGattUuids.auth) _authChar = c;
      if (c.uuid == HubGattUuids.command) _commandChar = c;
      if (c.uuid == HubGattUuids.statusFeed) _statusFeedChar = c;
      if (c.uuid == HubGattUuids.macList) _macListChar = c;
    }
  }

  Future<void> disconnect() async {
    await _statusFeedSub?.cancel();
    await _macListSub?.cancel();
    _statusFeedSub = null;
    _macListSub = null;
    await _device?.disconnect();
    _device = null;
    _authChar = null;
    _commandChar = null;
    _statusFeedChar = null;
    _macListChar = null;
  }

  // ---------------------------------------------------------------------------
  // Auth (§2: Write 12-byte ASCII code, Indicate result)
  // ---------------------------------------------------------------------------

  /// Authenticate with the hub using a 12-character code.
  /// Returns the auth result (accepted / rejected / timeout).
  Future<HubAuthResult> authenticate(String code) async {
    final char = _authChar;
    if (char == null) {
      throw StateError('Auth characteristic not available');
    }

    // Subscribe to indications before writing.
    final completer = Completer<HubAuthResult>();
    StreamSubscription<List<int>>? sub;

    sub = char.onValueReceived.listen((value) {
      sub?.cancel();
      if (value.isNotEmpty && value[0] == 0x01) {
        completer.complete(HubAuthResult.accepted);
      } else {
        completer.complete(HubAuthResult.rejected);
      }
    });

    await char.setNotifyValue(true);

    // Write the 12-byte ASCII code.
    final codeBytes = utf8.encode(code.padRight(12).substring(0, 12));
    await char.write(codeBytes, withoutResponse: false);

    // Wait for indication with timeout.
    final result = await completer.future.timeout(
      const Duration(seconds: 10),
      onTimeout: () {
        sub?.cancel();
        return HubAuthResult.timeout;
      },
    );

    return result;
  }

  // ---------------------------------------------------------------------------
  // Subscribe to status feed (§5, 1 Hz Notify)
  // ---------------------------------------------------------------------------

  Future<void> subscribeStatusFeed() async {
    final char = _statusFeedChar;
    if (char == null) {
      throw StateError('Status feed characteristic not available');
    }

    await char.setNotifyValue(true);
    _statusFeedSub = char.onValueReceived.listen(
      (bytes) {
        try {
          final status = HubStatus.fromUtf8Bytes(bytes);
          _statusController.add(status);
        } on FormatException {
          // Silently skip malformed frames.
        }
      },
    );
  }

  // ---------------------------------------------------------------------------
  // Subscribe to MAC list (§2, Read + Notify)
  // ---------------------------------------------------------------------------

  Future<void> subscribeMacList() async {
    final char = _macListChar;
    if (char == null) {
      throw StateError('MAC list characteristic not available');
    }

    await char.setNotifyValue(true);
    _macListSub = char.onValueReceived.listen(
      (bytes) {
        try {
          final pens = PenInfo.listFromBytes(bytes);
          _penListController.add(pens);
        } on FormatException {
          // Silently skip malformed frames.
        }
      },
    );
  }

  /// One-shot read of the current MAC list.
  Future<List<PenInfo>> readMacList() async {
    final char = _macListChar;
    if (char == null) {
      throw StateError('MAC list characteristic not available');
    }
    final bytes = await char.read();
    return PenInfo.listFromBytes(bytes);
  }

  // ---------------------------------------------------------------------------
  // Command writes (§4)
  // ---------------------------------------------------------------------------

  /// Write a command to the hub's command characteristic.
  Future<void> _writeCommand(
    int cmdId,
    String requestId,
    Map<String, dynamic> payload,
  ) async {
    final char = _commandChar;
    if (char == null) {
      throw StateError('Command characteristic not available');
    }

    final payloadJson = utf8.encode(jsonEncode(payload));
    final reqIdBytes =
        utf8.encode(requestId.padRight(16).substring(0, 16));

    final buffer = BytesBuilder();
    buffer.addByte(cmdId);
    buffer.add(reqIdBytes);
    buffer.add(payloadJson);

    await char.write(
      buffer.toBytes(),
      withoutResponse: false,
    );
  }

  /// Start an exam session (cmd 0x01).
  Future<void> startExam({
    required String examId,
    required int durationSec,
    required String requestId,
  }) =>
      _writeCommand(HubCommandId.startExam, requestId, {
        'exam_id': examId,
        'duration_sec': durationSec,
      });

  /// Stop a running exam (cmd 0x02).
  Future<void> stopExam({
    required String examId,
    String reason = 'manual',
    required String requestId,
  }) =>
      _writeCommand(HubCommandId.stopExam, requestId, {
        'exam_id': examId,
        'reason': reason,
      });

  /// Trigger pen registration scan on the hub (cmd 0x03).
  Future<void> startRegistrationScan({
    required String examId,
    required String requestId,
  }) =>
      _writeCommand(HubCommandId.startRegistrationScan, requestId, {
        'exam_id': examId,
      });

  /// Manually register a pen to a student (cmd 0x04).
  Future<void> manualRegister({
    required String examId,
    required String penMac,
    required String studentId,
    required String requestId,
  }) =>
      _writeCommand(HubCommandId.manualRegister, requestId, {
        'exam_id': examId,
        'pen_mac': penMac,
        'student_id': studentId,
      });

  /// Trigger stroke upload (cmd 0x05).
  Future<void> startUpload({
    required String examId,
    required String path,
    required String requestId,
  }) =>
      _writeCommand(HubCommandId.startUpload, requestId, {
        'exam_id': examId,
        'path': path,
      });

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  Future<void> dispose() async {
    await _statusFeedSub?.cancel();
    await _macListSub?.cancel();
    await _statusController.close();
    await _penListController.close();
    await _device?.disconnect();
  }
}
