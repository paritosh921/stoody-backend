/// Pen registration screen — scan trigger, pen list, manual register.
///
/// Displays pens discovered by the hub's BLE scan, showing MAC, RSSI,
/// battery, and binding status. Provides manual registration by
/// entering pen MAC + student ID.
library;

import 'dart:async';

import 'package:flutter/material.dart';

import '../ble_service.dart';
import '../models/pen_info.dart';
import '../widgets/pen_list_tile.dart';

class PenRegistrationScreen extends StatefulWidget {
  const PenRegistrationScreen({super.key});

  @override
  State<PenRegistrationScreen> createState() => _PenRegistrationScreenState();
}

class _PenRegistrationScreenState extends State<PenRegistrationScreen> {
  final BleService _ble = BleService();

  List<PenInfo> _pens = [];
  StreamSubscription<List<PenInfo>>? _penSub;

  bool _scanning = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _penSub = _ble.penListStream.listen((pens) {
      if (mounted) setState(() => _pens = pens);
    });
    _loadInitialList();
  }

  Future<void> _loadInitialList() async {
    try {
      final pens = await _ble.readMacList();
      if (mounted) setState(() => _pens = pens);
    } catch (_) {
      // Hub may not be ready yet; rely on notify stream.
    }
  }

  @override
  void dispose() {
    _penSub?.cancel();
    super.dispose();
  }

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  Future<void> _triggerScan() async {
    setState(() {
      _scanning = true;
      _error = null;
    });

    try {
      final requestId = DateTime.now().millisecondsSinceEpoch
          .toRadixString(36)
          .padLeft(16, '0');
      await _ble.startRegistrationScan(
        examId: '', // Hub uses current active exam.
        requestId: requestId,
      );
    } catch (e) {
      if (mounted) setState(() => _error = 'Scan trigger failed: $e');
    } finally {
      // The hub sends pen list updates via notify; scanning flag is
      // a local UI indicator — clear it after a short delay.
      await Future<void>.delayed(const Duration(seconds: 5));
      if (mounted) setState(() => _scanning = false);
    }
  }

  Future<void> _showManualRegisterDialog() async {
    final macController = TextEditingController();
    final studentIdController = TextEditingController();

    final result = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Manual Register'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: macController,
              decoration: const InputDecoration(
                labelText: 'Pen MAC Address',
                hintText: 'AA:BB:CC:DD:EE:FF',
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: studentIdController,
              decoration: const InputDecoration(
                labelText: 'Student ID',
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Register'),
          ),
        ],
      ),
    );

    if (result != true) return;

    final mac = macController.text.trim();
    final studentId = studentIdController.text.trim();
    if (mac.isEmpty || studentId.isEmpty) return;

    try {
      final requestId = DateTime.now().millisecondsSinceEpoch
          .toRadixString(36)
          .padLeft(16, '0');
      await _ble.manualRegister(
        examId: '',
        penMac: mac,
        studentId: studentId,
        requestId: requestId,
      );
    } catch (e) {
      if (mounted) setState(() => _error = 'Register failed: $e');
    }
  }

  // ---------------------------------------------------------------------------
  // Build
  // ---------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Pen Registration'),
        actions: [
          IconButton(
            icon: const Icon(Icons.person_add),
            tooltip: 'Manual Register',
            onPressed: _showManualRegisterDialog,
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _scanning ? null : _triggerScan,
        icon: _scanning
            ? const SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  color: Colors.white,
                ),
              )
            : const Icon(Icons.bluetooth_searching),
        label: Text(_scanning ? 'Scanning...' : 'Scan Pens'),
      ),
      body: Column(
        children: [
          if (_error != null)
            Container(
              width: double.infinity,
              color: Colors.red.shade50,
              padding: const EdgeInsets.all(12),
              child: Text(_error!, style: const TextStyle(color: Colors.red)),
            ),
          Expanded(
            child: _pens.isEmpty
                ? const Center(
                    child: Text('No pens discovered yet. Tap scan to begin.'),
                  )
                : ListView.separated(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    itemCount: _pens.length,
                    separatorBuilder: (_, __) => const Divider(height: 1),
                    itemBuilder: (context, index) =>
                        PenListTile(pen: _pens[index]),
                  ),
          ),
        ],
      ),
    );
  }
}
