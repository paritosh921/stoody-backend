/// Reusable sync progress bar widget.
///
/// A rounded linear progress indicator with optional label, used
/// by both the sync monitor and upload screens.
library;

import 'package:flutter/material.dart';

class SyncProgressBar extends StatelessWidget {
  const SyncProgressBar({
    super.key,
    required this.fraction,
    this.height = 10,
    this.label,
    this.activeColor,
    this.backgroundColor,
  });

  /// Progress fraction, clamped to 0.0 .. 1.0.
  final double fraction;

  /// Bar height in logical pixels.
  final double height;

  /// Optional label displayed to the right (e.g. "85%").
  final String? label;

  /// Override the filled portion colour.
  final Color? activeColor;

  /// Override the track colour.
  final Color? backgroundColor;

  @override
  Widget build(BuildContext context) {
    final clamped = fraction.clamp(0.0, 1.0);
    final pct = (clamped * 100).toStringAsFixed(0);
    final displayLabel = label ?? '$pct%';

    final Color barColor;
    if (activeColor != null) {
      barColor = activeColor!;
    } else if (clamped >= 1.0) {
      barColor = Colors.green;
    } else if (clamped >= 0.5) {
      barColor = Colors.blue;
    } else {
      barColor = Colors.orange;
    }

    return Row(
      children: [
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(height / 2),
            child: LinearProgressIndicator(
              value: clamped,
              minHeight: height,
              color: barColor,
              backgroundColor:
                  backgroundColor ?? Colors.grey.shade200,
            ),
          ),
        ),
        const SizedBox(width: 12),
        SizedBox(
          width: 40,
          child: Text(
            displayLabel,
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
            ),
            textAlign: TextAlign.end,
          ),
        ),
      ],
    );
  }
}
