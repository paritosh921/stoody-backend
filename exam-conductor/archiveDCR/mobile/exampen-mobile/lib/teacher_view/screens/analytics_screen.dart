/// Class analytics screen.
///
/// Shows aggregated statistics for an exam class: mean score, median,
/// standard deviation, score distribution histogram, and AI confidence
/// overview.  Data is derived from the class scores endpoint.
library;

import 'dart:math' as math;

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

import '../api/teacher_api.dart';

class AnalyticsScreen extends StatefulWidget {
  final String examId;
  final TeacherApi api;

  const AnalyticsScreen({
    super.key,
    required this.examId,
    required this.api,
  });

  @override
  State<AnalyticsScreen> createState() => _AnalyticsScreenState();
}

class _AnalyticsScreenState extends State<AnalyticsScreen> {
  List<ClassScoreRow> _rows = [];
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
      final rows = await widget.api.getClassScores(widget.examId);
      if (!mounted) return;
      setState(() {
        _rows = rows;
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

  // -- Derived stats --------------------------------------------------------

  double get _mean {
    if (_rows.isEmpty) return 0;
    return _rows.map((r) => r.totalScore).reduce((a, b) => a + b) /
        _rows.length;
  }

  double get _median {
    if (_rows.isEmpty) return 0;
    final sorted = _rows.map((r) => r.totalScore).toList()..sort();
    final mid = sorted.length ~/ 2;
    if (sorted.length.isOdd) return sorted[mid];
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }

  double get _stdDev {
    if (_rows.length < 2) return 0;
    final m = _mean;
    final variance =
        _rows.map((r) => math.pow(r.totalScore - m, 2)).reduce((a, b) => a + b) /
            _rows.length;
    return math.sqrt(variance);
  }

  double get _avgConfidence {
    if (_rows.isEmpty) return 0;
    return _rows.map((r) => r.aiConfidence).reduce((a, b) => a + b) /
        _rows.length;
  }

  double get _highScore {
    if (_rows.isEmpty) return 0;
    return _rows.map((r) => r.totalScore).reduce(math.max);
  }

  double get _lowScore {
    if (_rows.isEmpty) return 0;
    return _rows.map((r) => r.totalScore).reduce(math.min);
  }

  /// Score distribution grouped into 10-point buckets.
  Map<String, int> get _distribution {
    final buckets = <String, int>{};
    for (final row in _rows) {
      final bucket = (row.totalScore ~/ 10) * 10;
      final label = '$bucket-${bucket + 9}';
      buckets[label] = (buckets[label] ?? 0) + 1;
    }
    return Map.fromEntries(
      buckets.entries.toList()
        ..sort((a, b) => a.key.compareTo(b.key)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Class Analytics')),
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
    if (_rows.isEmpty) {
      return const Center(child: Text('No data available'));
    }

    final isTablet = MediaQuery.sizeOf(context).shortestSide >= 600;

    return RefreshIndicator(
      onRefresh: _load,
      child: ListView(
        padding: EdgeInsets.all(isTablet ? 24 : 16),
        children: [
          _buildStatGrid(isTablet),
          const SizedBox(height: 24),
          _buildDistributionChart(),
          const SizedBox(height: 24),
          _buildConfidenceOverview(),
        ],
      ),
    );
  }

  // -- Stat grid ------------------------------------------------------------

  Widget _buildStatGrid(bool isTablet) {
    final stats = [
      _StatData('Students', '${_rows.length}'),
      _StatData('Mean', _mean.toStringAsFixed(1)),
      _StatData('Median', _median.toStringAsFixed(1)),
      _StatData('Std Dev', _stdDev.toStringAsFixed(1)),
      _StatData('High', _highScore.toStringAsFixed(1)),
      _StatData('Low', _lowScore.toStringAsFixed(1)),
    ];

    final crossCount = isTablet ? 3 : 2;

    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: crossCount,
        mainAxisSpacing: 8,
        crossAxisSpacing: 8,
        childAspectRatio: 2.2,
      ),
      itemCount: stats.length,
      itemBuilder: (_, index) {
        final s = stats[index];
        return _StatCard(label: s.label, value: s.value);
      },
    );
  }

  // -- Distribution chart ---------------------------------------------------

  Widget _buildDistributionChart() {
    final theme = Theme.of(context);
    final dist = _distribution;
    if (dist.isEmpty) return const SizedBox.shrink();

    final entries = dist.entries.toList();
    final maxCount =
        entries.map((e) => e.value).reduce(math.max).toDouble();

    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Score Distribution',
              style: theme.textTheme.titleMedium,
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: BarChart(
                BarChartData(
                  alignment: BarChartAlignment.spaceAround,
                  maxY: maxCount + 1,
                  barTouchData: BarTouchData(enabled: true),
                  titlesData: FlTitlesData(
                    leftTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false),
                    ),
                    rightTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false),
                    ),
                    topTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false),
                    ),
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        getTitlesWidget: (value, _) {
                          final idx = value.toInt();
                          if (idx < 0 || idx >= entries.length) {
                            return const SizedBox.shrink();
                          }
                          return Padding(
                            padding: const EdgeInsets.only(top: 4),
                            child: Text(
                              entries[idx].key,
                              style: const TextStyle(fontSize: 9),
                            ),
                          );
                        },
                      ),
                    ),
                  ),
                  borderData: FlBorderData(show: false),
                  gridData: const FlGridData(show: false),
                  barGroups: List.generate(entries.length, (i) {
                    return BarChartGroupData(
                      x: i,
                      barRods: [
                        BarChartRodData(
                          toY: entries[i].value.toDouble(),
                          color: theme.colorScheme.primary,
                          width: 18,
                          borderRadius: const BorderRadius.vertical(
                            top: Radius.circular(4),
                          ),
                        ),
                      ],
                    );
                  }),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // -- Confidence overview --------------------------------------------------

  Widget _buildConfidenceOverview() {
    final theme = Theme.of(context);
    final avgPct = (_avgConfidence * 100).round();
    final lowConf = _rows.where((r) => r.aiConfidence < 0.5).length;

    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'AI Confidence Overview',
              style: theme.textTheme.titleMedium,
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: _StatCard(
                    label: 'Avg Confidence',
                    value: '$avgPct%',
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _StatCard(
                    label: 'Low Confidence',
                    value: '$lowConf student(s)',
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

class _StatData {
  final String label;
  final String value;
  const _StatData(this.label, this.value);
}

class _StatCard extends StatelessWidget {
  final String label;
  final String value;

  const _StatCard({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      elevation: 0,
      color: theme.colorScheme.surfaceContainerHighest,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
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
