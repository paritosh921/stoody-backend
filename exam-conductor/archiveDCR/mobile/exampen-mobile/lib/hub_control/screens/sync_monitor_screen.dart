/// Sync monitor screen — per-pen sync progress, dongle health, totals.
///
/// Combines the 1 Hz status feed (aggregate sync counters) with the
/// MAC list (per-pen detail) to show a live sync dashboard.
library;

import 'dart:async';

import 'package:flutter/material.dart';

import '../ble_service.dart';
import '../models/hub_status.dart';
import '../models/pen_info.dart';
import '../models/sync_progress.dart';
import '../widgets/sync_progress_bar.dart';

class SyncMonitorScreen extends StatefulWidget {
  const SyncMonitorScreen({super.key});

  @override
  State<SyncMonitorScreen> createState() => _SyncMonitorScreenState();
}

class _SyncMonitorScreenState extends State<SyncMonitorScreen> {
  final BleService _ble = BleService();

  HubStatus? _status;
  List<PenInfo> _pens = [];

  StreamSubscription<HubStatus>? _statusSub;
  StreamSubscription<List<PenInfo>>? _penSub;

  @override
  void initState() {
    super.initState();
    _statusSub = _ble.statusStream.listen((s) {
      if (mounted) setState(() => _status = s);
    });
    _penSub = _ble.penListStream.listen((p) {
      if (mounted) setState(() => _pens = p);
    });
  }

  @override
  void dispose() {
    _statusSub?.cancel();
    _penSub?.cancel();
    super.dispose();
  }

  // ---------------------------------------------------------------------------
  // Build
  // ---------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    final syncSummary = _status?.sync;

    return Scaffold(
      appBar: AppBar(title: const Text('Sync Monitor')),
      body: Column(
        children: [
          // Aggregate progress header
          _buildAggregateHeader(syncSummary),
          const Divider(height: 1),

          // Per-pen list
          Expanded(
            child: _pens.isEmpty
                ? const Center(child: Text('No pen data available.'))
                : ListView.builder(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    itemCount: _pens.length,
                    itemBuilder: (context, index) =>
                        _buildPenSyncRow(_pens[index]),
                  ),
          ),
        ],
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Aggregate header
  // ---------------------------------------------------------------------------

  Widget _buildAggregateHeader(SyncSummary? sync) {
    if (sync == null) {
      return const Padding(
        padding: EdgeInsets.all(24),
        child: Text(
          'Waiting for hub status...',
          textAlign: TextAlign.center,
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          SyncProgressBar(
            fraction: sync.progressFraction,
            height: 14,
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _counter('Complete', sync.complete, Colors.green),
              _counter('In Progress', sync.inProgress, Colors.blue),
              _counter('Failed', sync.failed, Colors.red),
              _counter('Pending', sync.pending, Colors.grey),
            ],
          ),
        ],
      ),
    );
  }

  Widget _counter(String label, int count, Color color) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '$count',
          style: TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
        Text(label, style: const TextStyle(fontSize: 11)),
      ],
    );
  }

  // ---------------------------------------------------------------------------
  // Per-pen row
  // ---------------------------------------------------------------------------

  Widget _buildPenSyncRow(PenInfo pen) {
    // Derive a coarse sync state from pen binding status.
    final PenSyncState syncState;
    switch (pen.status) {
      case PenBindingStatus.confirmed:
        syncState = PenSyncState.complete;
      case PenBindingStatus.rejected:
        syncState = PenSyncState.failed;
      case PenBindingStatus.provisional:
        syncState = PenSyncState.syncing;
      case PenBindingStatus.discovered:
        syncState = PenSyncState.pending;
    }

    final Color stateColor;
    switch (syncState) {
      case PenSyncState.complete:
        stateColor = Colors.green;
      case PenSyncState.syncing:
      case PenSyncState.connecting:
        stateColor = Colors.blue;
      case PenSyncState.failed:
      case PenSyncState.timeout:
        stateColor = Colors.red;
      case PenSyncState.pending:
        stateColor = Colors.grey;
    }

    return ListTile(
      leading: CircleAvatar(
        backgroundColor: stateColor.withValues(alpha: 0.15),
        child: Icon(Icons.edit, color: stateColor, size: 20),
      ),
      title: Text(pen.mac),
      subtitle: Row(
        children: [
          Text(syncState.name.toUpperCase(),
              style: TextStyle(color: stateColor, fontSize: 12)),
          if (pen.batteryPct != null) ...[
            const SizedBox(width: 12),
            Icon(Icons.battery_std, size: 14, color: Colors.grey.shade600),
            Text(' ${pen.batteryLabel}',
                style: const TextStyle(fontSize: 12)),
          ],
        ],
      ),
      trailing: pen.studentName != null
          ? Text(pen.studentName!, style: const TextStyle(fontSize: 12))
          : null,
    );
  }
}
