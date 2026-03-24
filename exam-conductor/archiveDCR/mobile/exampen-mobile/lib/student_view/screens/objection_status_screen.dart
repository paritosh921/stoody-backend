/// Objection status tracking screen.
///
/// Lists all objections filed by the current student with status indicators,
/// resolution details, and updated scores when resolved.
library;

import 'package:flutter/material.dart';

import '../api/student_api.dart';

class ObjectionStatusScreen extends StatefulWidget {
  final StudentApi api;

  const ObjectionStatusScreen({super.key, required this.api});

  @override
  State<ObjectionStatusScreen> createState() => _ObjectionStatusScreenState();
}

class _ObjectionStatusScreenState extends State<ObjectionStatusScreen> {
  List<StudentObjection> _objections = [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final objections = await widget.api.listObjections();
      if (!mounted) return;
      setState(() {
        _objections = objections;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('My Objections')),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return _ErrorRetry(message: _error!, onRetry: _load);
    }
    if (_objections.isEmpty) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.check_circle_outline, size: 48, color: Colors.grey),
            SizedBox(height: 12),
            Text('No objections filed'),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _load,
      child: ListView.separated(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        itemCount: _objections.length,
        separatorBuilder: (_, __) => const SizedBox(height: 8),
        itemBuilder: (_, index) {
          return _ObjectionCard(objection: _objections[index]);
        },
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Objection card
// ---------------------------------------------------------------------------

class _ObjectionCard extends StatelessWidget {
  final StudentObjection objection;

  const _ObjectionCard({required this.objection});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isTablet = MediaQuery.sizeOf(context).shortestSide >= 600;

    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: EdgeInsets.all(isTablet ? 20 : 14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header row
            Row(
              children: [
                _StatusIcon(status: objection.status),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Question ${objection.questionId}',
                        style: theme.textTheme.titleSmall,
                      ),
                      Text(
                        'Exam: ${_shortenId(objection.examId)}',
                        style: theme.textTheme.bodySmall,
                      ),
                    ],
                  ),
                ),
                _StatusBadge(status: objection.status),
              ],
            ),

            // Status timeline
            const SizedBox(height: 12),
            _StatusTimeline(currentStatus: objection.status),

            // Objection text
            if (objection.objectionText != null &&
                objection.objectionText!.isNotEmpty) ...[
              const Divider(height: 20),
              Text(
                objection.objectionText!,
                style: theme.textTheme.bodyMedium,
                maxLines: 3,
                overflow: TextOverflow.ellipsis,
              ),
            ],

            // Resolution
            if (objection.status == ObjectionStatus.resolved) ...[
              const Divider(height: 20),
              _ResolutionBox(objection: objection),
            ],
          ],
        ),
      ),
    );
  }

  String _shortenId(String id) {
    return id.length > 8 ? '${id.substring(0, 8)}...' : id;
  }
}

// ---------------------------------------------------------------------------
// Status timeline
// ---------------------------------------------------------------------------

class _StatusTimeline extends StatelessWidget {
  final ObjectionStatus currentStatus;

  const _StatusTimeline({required this.currentStatus});

  static const _steps = [
    ObjectionStatus.filed,
    ObjectionStatus.assigned,
    ObjectionStatus.reviewing,
    ObjectionStatus.resolved,
  ];

  @override
  Widget build(BuildContext context) {
    final currentIdx = _steps.indexOf(currentStatus);
    final isEscalated = currentStatus == ObjectionStatus.escalated;

    return Row(
      children: List.generate(_steps.length * 2 - 1, (i) {
        if (i.isOdd) {
          // Connector line
          final stepBefore = i ~/ 2;
          final reached = stepBefore < currentIdx;
          return Expanded(
            child: Container(
              height: 2,
              color: reached
                  ? Theme.of(context).colorScheme.primary
                  : Colors.grey.shade300,
            ),
          );
        }

        final stepIdx = i ~/ 2;
        final reached = stepIdx <= currentIdx;
        final isCurrent = stepIdx == currentIdx;

        return Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: isCurrent ? 14 : 10,
              height: isCurrent ? 14 : 10,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: isEscalated && isCurrent
                    ? Colors.orange
                    : reached
                        ? Theme.of(context).colorScheme.primary
                        : Colors.grey.shade300,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              _steps[stepIdx].displayLabel,
              style: TextStyle(
                fontSize: 9,
                fontWeight: isCurrent ? FontWeight.bold : FontWeight.normal,
                color: reached ? null : Colors.grey,
              ),
            ),
          ],
        );
      }),
    );
  }
}

// ---------------------------------------------------------------------------
// Resolution box
// ---------------------------------------------------------------------------

class _ResolutionBox extends StatelessWidget {
  final StudentObjection objection;

  const _ResolutionBox({required this.objection});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.green.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.green.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Resolution',
            style: theme.textTheme.labelMedium?.copyWith(
              color: Colors.green.shade800,
            ),
          ),
          if (objection.resolutionReason != null) ...[
            const SizedBox(height: 4),
            Text(objection.resolutionReason!),
          ],
          if (objection.newScore != null) ...[
            const SizedBox(height: 4),
            Text(
              'Updated Score: ${objection.newScore!.toStringAsFixed(1)}',
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
          ],
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Status widgets
// ---------------------------------------------------------------------------

class _StatusIcon extends StatelessWidget {
  final ObjectionStatus status;
  const _StatusIcon({required this.status});

  @override
  Widget build(BuildContext context) {
    final (IconData icon, Color color) = switch (status) {
      ObjectionStatus.filed => (Icons.send, Colors.blue),
      ObjectionStatus.assigned => (Icons.person_add, Colors.indigo),
      ObjectionStatus.reviewing => (Icons.rate_review, Colors.orange),
      ObjectionStatus.resolved => (Icons.check_circle, Colors.green),
      ObjectionStatus.escalated => (Icons.warning, Colors.deepOrange),
    };

    return CircleAvatar(
      radius: 18,
      backgroundColor: color.withValues(alpha: 0.15),
      child: Icon(icon, color: color, size: 20),
    );
  }
}

class _StatusBadge extends StatelessWidget {
  final ObjectionStatus status;
  const _StatusBadge({required this.status});

  @override
  Widget build(BuildContext context) {
    final (Color bg, Color fg) = switch (status) {
      ObjectionStatus.filed => (Colors.blue.shade50, Colors.blue.shade800),
      ObjectionStatus.assigned =>
        (Colors.indigo.shade50, Colors.indigo.shade800),
      ObjectionStatus.reviewing =>
        (Colors.orange.shade50, Colors.orange.shade800),
      ObjectionStatus.resolved =>
        (Colors.green.shade50, Colors.green.shade800),
      ObjectionStatus.escalated =>
        (Colors.deepOrange.shade50, Colors.deepOrange.shade800),
    };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        status.displayLabel,
        style: TextStyle(
          color: fg,
          fontSize: 11,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Error + retry
// ---------------------------------------------------------------------------

class _ErrorRetry extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;

  const _ErrorRetry({required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.error_outline,
                size: 48, color: Theme.of(context).colorScheme.error),
            const SizedBox(height: 12),
            Text(message, textAlign: TextAlign.center),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh),
              label: const Text('Retry'),
            ),
          ],
        ),
      ),
    );
  }
}
