/// Leaderboard screen.
///
/// Displays a ranked list of students for a given exam, sorted by total score
/// descending.  The top 3 students receive medal indicators.  Data is sourced
/// from the class scores endpoint.
library;

import 'package:flutter/material.dart';

import '../api/teacher_api.dart';

class LeaderboardScreen extends StatefulWidget {
  final String examId;
  final TeacherApi api;

  const LeaderboardScreen({
    super.key,
    required this.examId,
    required this.api,
  });

  @override
  State<LeaderboardScreen> createState() => _LeaderboardScreenState();
}

class _LeaderboardScreenState extends State<LeaderboardScreen> {
  List<ClassScoreRow> _ranked = [];
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
      rows.sort((a, b) => b.totalScore.compareTo(a.totalScore));
      if (!mounted) return;
      setState(() {
        _ranked = rows;
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
      appBar: AppBar(title: const Text('Leaderboard')),
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
    if (_ranked.isEmpty) {
      return const Center(child: Text('No scores available'));
    }
    return RefreshIndicator(
      onRefresh: _load,
      child: _buildList(),
    );
  }

  Widget _buildList() {
    final isTablet = MediaQuery.sizeOf(context).shortestSide >= 600;

    return ListView.builder(
      padding: EdgeInsets.symmetric(
        horizontal: isTablet ? 32 : 16,
        vertical: 12,
      ),
      itemCount: _ranked.length,
      itemBuilder: (context, index) {
        final row = _ranked[index];
        final rank = index + 1;
        return _LeaderboardTile(row: row, rank: rank);
      },
    );
  }
}

// ---------------------------------------------------------------------------
// Leaderboard tile
// ---------------------------------------------------------------------------

class _LeaderboardTile extends StatelessWidget {
  final ClassScoreRow row;
  final int rank;

  const _LeaderboardTile({required this.row, required this.rank});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isTop3 = rank <= 3;

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      elevation: isTop3 ? 2 : 1,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: isTop3
            ? BorderSide(color: _medalColor, width: 1.5)
            : BorderSide.none,
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        child: Row(
          children: [
            // Rank badge
            SizedBox(
              width: 40,
              child: isTop3
                  ? _MedalBadge(rank: rank)
                  : Text(
                      '#$rank',
                      textAlign: TextAlign.center,
                      style: theme.textTheme.titleSmall?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    ),
            ),
            const SizedBox(width: 12),

            // Student name
            Expanded(
              child: Text(
                row.studentName,
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: isTop3 ? FontWeight.bold : FontWeight.normal,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),

            // Score
            Text(
              row.totalScore.toStringAsFixed(1),
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
                color: isTop3 ? _medalColor : null,
              ),
            ),

            // Percentile (if available)
            if (row.percentile != null) ...[
              const SizedBox(width: 12),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: theme.colorScheme.secondaryContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'P${row.percentile!.round()}',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: theme.colorScheme.onSecondaryContainer,
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Color get _medalColor => switch (rank) {
        1 => const Color(0xFFFFD700), // gold
        2 => const Color(0xFFC0C0C0), // silver
        3 => const Color(0xFFCD7F32), // bronze
        _ => Colors.grey,
      };
}

// ---------------------------------------------------------------------------
// Medal badge
// ---------------------------------------------------------------------------

class _MedalBadge extends StatelessWidget {
  final int rank;
  const _MedalBadge({required this.rank});

  @override
  Widget build(BuildContext context) {
    final (IconData icon, Color color) = switch (rank) {
      1 => (Icons.emoji_events, const Color(0xFFFFD700)),
      2 => (Icons.emoji_events, const Color(0xFFC0C0C0)),
      3 => (Icons.emoji_events, const Color(0xFFCD7F32)),
      _ => (Icons.tag, Colors.grey),
    };

    return Icon(icon, color: color, size: 28);
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
