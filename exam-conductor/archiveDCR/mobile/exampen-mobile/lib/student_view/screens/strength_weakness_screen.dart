/// AI-generated strength and weakness analysis screen.
///
/// Displays strengths and weaknesses extracted from the performance endpoint.
/// Presented as categorized lists with visual indicators.
library;

import 'package:flutter/material.dart';

import '../api/student_api.dart';

class StrengthWeaknessScreen extends StatefulWidget {
  final StudentApi api;

  const StrengthWeaknessScreen({super.key, required this.api});

  @override
  State<StrengthWeaknessScreen> createState() =>
      _StrengthWeaknessScreenState();
}

class _StrengthWeaknessScreenState extends State<StrengthWeaknessScreen> {
  PerformanceView? _performance;
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
      final perf = await widget.api.getPerformanceHistory();
      if (!mounted) return;
      setState(() {
        _performance = perf;
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
      appBar: AppBar(title: const Text('Strengths & Weaknesses')),
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

    final perf = _performance!;
    final isTablet = MediaQuery.sizeOf(context).shortestSide >= 600;

    if (perf.strengths.isEmpty && perf.weaknesses.isEmpty) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(32),
          child: Text(
            'No analysis available yet. '
            'Complete more exams to see AI-generated insights.',
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    if (isTablet) {
      return _buildTabletLayout(perf);
    }
    return _buildPhoneLayout(perf);
  }

  // -- Phone: vertical stack ------------------------------------------------

  Widget _buildPhoneLayout(PerformanceView perf) {
    return RefreshIndicator(
      onRefresh: _load,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          if (perf.strengths.isNotEmpty) ...[
            _SectionCard(
              title: 'Strengths',
              items: perf.strengths,
              icon: Icons.thumb_up_alt_outlined,
              color: Colors.green,
            ),
            const SizedBox(height: 16),
          ],
          if (perf.weaknesses.isNotEmpty)
            _SectionCard(
              title: 'Areas for Improvement',
              items: perf.weaknesses,
              icon: Icons.trending_up,
              color: Colors.orange,
            ),
        ],
      ),
    );
  }

  // -- Tablet: side-by-side -------------------------------------------------

  Widget _buildTabletLayout(PerformanceView perf) {
    return RefreshIndicator(
      onRefresh: _load,
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (perf.strengths.isNotEmpty)
              Expanded(
                child: _SectionCard(
                  title: 'Strengths',
                  items: perf.strengths,
                  icon: Icons.thumb_up_alt_outlined,
                  color: Colors.green,
                ),
              ),
            if (perf.strengths.isNotEmpty && perf.weaknesses.isNotEmpty)
              const SizedBox(width: 16),
            if (perf.weaknesses.isNotEmpty)
              Expanded(
                child: _SectionCard(
                  title: 'Areas for Improvement',
                  items: perf.weaknesses,
                  icon: Icons.trending_up,
                  color: Colors.orange,
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Section card
// ---------------------------------------------------------------------------

class _SectionCard extends StatelessWidget {
  final String title;
  final List<String> items;
  final IconData icon;
  final Color color;

  const _SectionCard({
    required this.title,
    required this.items,
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Section header
            Row(
              children: [
                CircleAvatar(
                  radius: 18,
                  backgroundColor: color.withValues(alpha: 0.15),
                  child: Icon(icon, color: color, size: 20),
                ),
                const SizedBox(width: 12),
                Text(title, style: theme.textTheme.titleMedium),
              ],
            ),
            const SizedBox(height: 16),

            // Items
            ...items.asMap().entries.map((entry) {
              return Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      margin: const EdgeInsets.only(top: 6),
                      width: 8,
                      height: 8,
                      decoration: BoxDecoration(
                        color: color,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        entry.value,
                        style: theme.textTheme.bodyMedium,
                      ),
                    ),
                  ],
                ),
              );
            }),
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
