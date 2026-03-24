/// Reusable pen info tile — shows MAC, RSSI, battery, binding status.
///
/// Used in pen registration and sync monitor screens.
library;

import 'package:flutter/material.dart';

import '../models/pen_info.dart';

class PenListTile extends StatelessWidget {
  const PenListTile({
    super.key,
    required this.pen,
    this.onTap,
    this.trailing,
  });

  final PenInfo pen;
  final VoidCallback? onTap;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      onTap: onTap,
      leading: _buildLeadingIcon(),
      title: Text(
        pen.mac,
        style: const TextStyle(fontFamily: 'monospace', fontSize: 14),
      ),
      subtitle: _buildSubtitle(),
      trailing: trailing ?? _buildTrailing(),
    );
  }

  // ---------------------------------------------------------------------------
  // Sub-widgets
  // ---------------------------------------------------------------------------

  Widget _buildLeadingIcon() {
    final Color statusColor;
    switch (pen.status) {
      case PenBindingStatus.confirmed:
        statusColor = Colors.green;
      case PenBindingStatus.provisional:
        statusColor = Colors.orange;
      case PenBindingStatus.rejected:
        statusColor = Colors.red;
      case PenBindingStatus.discovered:
        statusColor = Colors.grey;
    }

    return CircleAvatar(
      backgroundColor: statusColor.withValues(alpha: 0.15),
      child: Icon(Icons.edit, color: statusColor, size: 20),
    );
  }

  Widget _buildSubtitle() {
    final parts = <Widget>[];

    // Binding status
    parts.add(
      Text(
        pen.status.name.toUpperCase(),
        style: TextStyle(
          fontSize: 11,
          color: _statusTextColor(),
          fontWeight: FontWeight.w600,
        ),
      ),
    );

    // RSSI
    if (pen.rssi != null) {
      parts.add(const SizedBox(width: 10));
      parts.add(
        Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.signal_cellular_alt, size: 13, color: _rssiColor()),
            Text(' ${pen.rssi} dBm', style: const TextStyle(fontSize: 11)),
          ],
        ),
      );
    }

    // Battery
    if (pen.batteryPct != null) {
      parts.add(const SizedBox(width: 10));
      parts.add(
        Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              _batteryIcon(),
              size: 14,
              color: _batteryColor(),
            ),
            Text(' ${pen.batteryLabel}', style: const TextStyle(fontSize: 11)),
          ],
        ),
      );
    }

    return Row(children: parts);
  }

  Widget? _buildTrailing() {
    if (pen.studentName != null) {
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Text(
            pen.studentName!,
            style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500),
          ),
          if (pen.studentRoll != null)
            Text(
              pen.studentRoll!,
              style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
            ),
        ],
      );
    }
    return null;
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  Color _statusTextColor() {
    switch (pen.status) {
      case PenBindingStatus.confirmed:
        return Colors.green.shade700;
      case PenBindingStatus.provisional:
        return Colors.orange.shade700;
      case PenBindingStatus.rejected:
        return Colors.red.shade700;
      case PenBindingStatus.discovered:
        return Colors.grey.shade600;
    }
  }

  Color _rssiColor() {
    final rssi = pen.rssi ?? -100;
    if (rssi >= -50) return Colors.green;
    if (rssi >= -70) return Colors.orange;
    return Colors.red;
  }

  IconData _batteryIcon() {
    final pct = pen.batteryPct ?? 0;
    if (pct >= 80) return Icons.battery_full;
    if (pct >= 50) return Icons.battery_5_bar;
    if (pct >= 20) return Icons.battery_3_bar;
    return Icons.battery_alert;
  }

  Color _batteryColor() {
    final pct = pen.batteryPct ?? 0;
    if (pct >= 50) return Colors.green;
    if (pct >= 20) return Colors.orange;
    return Colors.red;
  }
}
