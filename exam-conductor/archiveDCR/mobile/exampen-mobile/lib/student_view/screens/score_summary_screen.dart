/// Student score summary screen.
///
/// Displays the overall exam result: total score, percentage, percentile,
/// and pass/fail status.  A prominent visual indicator shows the result at
/// a glance.  Tapping "View Breakdown" navigates to the per-question screen.
library;

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../api/student_api.dart';

class ScoreSummaryScreen extends StatefulWidget {
  final String examId;
  final StudentApi api;

  const ScoreSummaryScreen({
    super.key,
    required this.examId,
    required this.api,
  });

  @override
  State<ScoreSummaryScreen> createState() => _ScoreSummaryScreenState();
}

class _ScoreSummaryScreenState extends State<ScoreSummaryScreen> {
  StudentScoreView? _score;
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
      final score = await widget.api.getScores(widget.examId);
      if (!mounted) return;
      setState(() {
        _score = score;
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
      appBar: AppBar(title: const Text('Score Summary')),
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

    final score = _score!;
    final isTablet = MediaQuery.sizeOf(context).shortestSide >= 600;

    return RefreshIndicator(
      onRefresh: _load,
      child: ListView(
        padding: EdgeInsets.all(isTablet ? 32 : 16),
        children: [
          _PassFailBanner(score: score),
          const SizedBox(height: 24),
          _ScoreRing(percentage: score.percentage),
          const SizedBox(height: 24),
          _StatRow(score: score),
          const SizedBox(height: 32),
          FilledButton.icon(
            onPressed: () {
              context.go('/student/exams/${widget.examId}/breakdown');
            },
            icon: const Icon(Icons.list_alt),
            label: const Text('View Question Breakdown'),
            style: FilledButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 16),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Pass / fail banner
// ---------------------------------------------------------------------------

class _PassFailBanner extends StatelessWidget {
  final StudentScoreView score;

  const _PassFailBanner({required this.score});

  @override
  Widget build(BuildContext context) {
    if (score.passFail == null) return const SizedBox.shrink();

    final passed = score.passed;
    final theme = Theme.of(context);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 14),
      decoration: BoxDecoration(
        color: passed
            ? Colors.green.withValues(alpha: 0.1)
            : Colors.red.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: passed ? Colors.green : Colors.red,
          width: 1.5,
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            passed ? Icons.check_circle : Icons.cancel,
            color: passed ? Colors.green : Colors.red,
          ),
          const SizedBox(width: 8),
          Text(
            passed ? 'PASS' : 'FAIL',
            style: theme.textTheme.titleLarge?.copyWith(
              color: passed ? Colors.green : Colors.red,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Score ring
// ---------------------------------------------------------------------------

class _ScoreRing extends StatelessWidget {
  final double percentage;

  const _ScoreRing({required this.percentage});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final pct = percentage.clamp(0, 100);
    final color = pct >= 60
        ? Colors.green
        : pct >= 40
            ? Colors.orange
            : Colors.red;

    return Center(
      child: SizedBox(
        width: 160,
        height: 160,
        child: Stack(
          fit: StackFit.expand,
          children: [
            CircularProgressIndicator(
              value: pct / 100,
              strokeWidth: 12,
              backgroundColor: color.withValues(alpha: 0.15),
              valueColor: AlwaysStoppedAnimation(color),
            ),
            Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    '${pct.toStringAsFixed(1)}%',
                    style: theme.textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: color,
                    ),
                  ),
                  Text('Score', style: theme.textTheme.bodySmall),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Stat row
// ---------------------------------------------------------------------------

class _StatRow extends StatelessWidget {
  final StudentScoreView score;

  const _StatRow({required this.score});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isTablet = MediaQuery.sizeOf(context).shortestSide >= 600;

    return Row(
      children: [
        Expanded(
          child: _StatCard(
            label: 'Total Score',
            value: score.totalScore.toStringAsFixed(1),
            isTablet: isTablet,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _StatCard(
            label: 'Percentage',
            value: '${score.percentage.toStringAsFixed(1)}%',
            isTablet: isTablet,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _StatCard(
            label: 'Percentile',
            value: 'P${score.percentile.round()}',
            isTablet: isTablet,
          ),
        ),
      ],
    );
  }
}

class _StatCard extends StatelessWidget {
  final String label;
  final String value;
  final bool isTablet;

  const _StatCard({
    required this.label,
    required this.value,
    this.isTablet = false,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      elevation: 0,
      color: theme.colorScheme.surfaceContainerHighest,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: EdgeInsets.symmetric(
          horizontal: 12,
          vertical: isTablet ? 16 : 12,
        ),
        child: Column(
          children: [
            Text(
              value,
              style: theme.textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              label,
              style: theme.textTheme.bodySmall,
              textAlign: TextAlign.center,
            ),
          ],
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
