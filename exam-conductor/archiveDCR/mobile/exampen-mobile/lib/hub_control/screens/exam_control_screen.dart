/// Exam control screen — start/stop exam, timer display, arm confirmation.
///
/// Listens to the BLE status feed for real-time timer and state updates.
/// Start/stop commands are written to the hub's command characteristic
/// with an idempotent request ID.
library;

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../ble_service.dart';
import '../models/hub_status.dart';

class ExamControlScreen extends StatefulWidget {
  const ExamControlScreen({super.key});

  @override
  State<ExamControlScreen> createState() => _ExamControlScreenState();
}

class _ExamControlScreenState extends State<ExamControlScreen> {
  final BleService _ble = BleService();

  HubStatus? _status;
  StreamSubscription<HubStatus>? _statusSub;

  bool _starting = false;
  bool _stopping = false;
  String? _error;

  // Exam config — populated before start.
  final TextEditingController _examIdController = TextEditingController();
  final TextEditingController _durationController =
      TextEditingController(text: '60');

  @override
  void initState() {
    super.initState();
    _statusSub = _ble.statusStream.listen((status) {
      if (mounted) setState(() => _status = status);
    });
  }

  @override
  void dispose() {
    _statusSub?.cancel();
    _examIdController.dispose();
    _durationController.dispose();
    super.dispose();
  }

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  Future<void> _startExam() async {
    final examId = _examIdController.text.trim();
    if (examId.isEmpty) {
      setState(() => _error = 'Exam ID is required');
      return;
    }
    final duration = int.tryParse(_durationController.text.trim());
    if (duration == null || duration <= 0) {
      setState(() => _error = 'Enter a valid duration in minutes');
      return;
    }

    // Arm confirmation.
    final confirmed = await _showArmDialog();
    if (!confirmed) return;

    setState(() {
      _starting = true;
      _error = null;
    });

    try {
      final requestId = _makeRequestId();
      await _ble.startExam(
        examId: examId,
        durationSec: duration * 60,
        requestId: requestId,
      );
    } catch (e) {
      if (mounted) setState(() => _error = 'Start failed: $e');
    } finally {
      if (mounted) setState(() => _starting = false);
    }
  }

  Future<void> _stopExam() async {
    final examId = _status?.examId ?? _examIdController.text.trim();
    if (examId.isEmpty) return;

    final confirmed = await _showStopDialog();
    if (!confirmed) return;

    setState(() {
      _stopping = true;
      _error = null;
    });

    try {
      final requestId = _makeRequestId();
      await _ble.stopExam(
        examId: examId,
        requestId: requestId,
      );
    } catch (e) {
      if (mounted) setState(() => _error = 'Stop failed: $e');
    } finally {
      if (mounted) setState(() => _stopping = false);
    }
  }

  String _makeRequestId() =>
      DateTime.now().millisecondsSinceEpoch.toRadixString(36).padLeft(16, '0');

  // ---------------------------------------------------------------------------
  // Dialogs
  // ---------------------------------------------------------------------------

  Future<bool> _showArmDialog() async {
    final result = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Start Exam'),
        content: const Text(
          'Are you sure you want to start the exam? '
          'This will begin the countdown timer on the hub.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Start'),
          ),
        ],
      ),
    );
    return result ?? false;
  }

  Future<bool> _showStopDialog() async {
    final result = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Stop Exam'),
        content: const Text(
          'Are you sure you want to stop the exam early? '
          'Pen sync will begin immediately.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red,
              foregroundColor: Colors.white,
            ),
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Stop'),
          ),
        ],
      ),
    );
    return result ?? false;
  }

  // ---------------------------------------------------------------------------
  // Build
  // ---------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    final isRunning =
        _status != null && _status!.state != 'idle' && _status!.state != 'unknown';

    return Scaffold(
      appBar: AppBar(
        title: const Text('Exam Control'),
        actions: [
          IconButton(
            icon: const Icon(Icons.app_registration),
            tooltip: 'Pen Registration',
            onPressed: () => context.go('/hub/register'),
          ),
          IconButton(
            icon: const Icon(Icons.sync),
            tooltip: 'Sync Monitor',
            onPressed: () => context.go('/hub/sync'),
          ),
          IconButton(
            icon: const Icon(Icons.upload),
            tooltip: 'Upload',
            onPressed: () => context.go('/hub/upload'),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Timer display
            _buildTimerCard(),
            const SizedBox(height: 24),

            // Exam ID & duration (editable only before start)
            if (!isRunning) ...[
              TextField(
                controller: _examIdController,
                decoration: const InputDecoration(
                  labelText: 'Exam ID',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _durationController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Duration (minutes)',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 24),
            ],

            // Action buttons
            if (!isRunning)
              ElevatedButton.icon(
                onPressed: _starting ? null : _startExam,
                icon: _starting
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.play_arrow),
                label: const Text('Start Exam'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
              )
            else
              ElevatedButton.icon(
                onPressed: _stopping ? null : _stopExam,
                icon: _stopping
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.stop),
                label: const Text('Stop Exam'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.red,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
              ),

            if (_error != null) ...[
              const SizedBox(height: 16),
              Text(
                _error!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.red),
              ),
            ],

            const Spacer(),

            // Status bar
            if (_status != null) _buildStatusBar(),
          ],
        ),
      ),
    );
  }

  Widget _buildTimerCard() {
    final display = _status?.timerDisplay ?? '--:--';
    final state = _status?.state ?? 'Disconnected';

    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 32, horizontal: 16),
        child: Column(
          children: [
            Text(
              display,
              style: Theme.of(context).textTheme.displayLarge?.copyWith(
                    fontFamily: 'monospace',
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 8),
            Chip(label: Text(state.toUpperCase())),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusBar() {
    final s = _status!;
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        _statusChip(
          icon: Icons.wifi,
          label: s.wifi.connected ? 'WiFi OK' : 'WiFi OFF',
          ok: s.wifi.connected,
        ),
        _statusChip(
          icon: Icons.sd_storage,
          label: s.storage.sdOk ? 'SD OK' : 'SD ERR',
          ok: s.storage.sdOk,
        ),
        _statusChip(
          icon: Icons.usb,
          label: s.storage.usbOk ? 'USB OK' : 'USB ERR',
          ok: s.storage.usbOk,
        ),
        _statusChip(
          icon: Icons.sync,
          label: '${s.sync.complete}/${s.sync.total}',
          ok: s.sync.failed == 0,
        ),
      ],
    );
  }

  Widget _statusChip({
    required IconData icon,
    required String label,
    required bool ok,
  }) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, color: ok ? Colors.green : Colors.red, size: 20),
        const SizedBox(height: 4),
        Text(label, style: const TextStyle(fontSize: 11)),
      ],
    );
  }
}
