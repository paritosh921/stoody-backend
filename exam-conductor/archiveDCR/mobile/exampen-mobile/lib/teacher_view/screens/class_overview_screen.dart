/// Class score overview screen.
///
/// Displays a sortable, filterable list of [ClassScoreRow] items for a
/// given exam.  Each row shows student name, total score, and AI confidence.
/// Tapping a row navigates to [StudentDetailScreen].
library;

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../api/teacher_api.dart';

// ---------------------------------------------------------------------------
// Sort criteria
// ---------------------------------------------------------------------------

enum _SortField { name, score, confidence }

// ---------------------------------------------------------------------------
// Screen
// ---------------------------------------------------------------------------

class ClassOverviewScreen extends StatefulWidget {
  final String examId;
  final TeacherApi api;

  const ClassOverviewScreen({
    super.key,
    required this.examId,
    required this.api,
  });

  @override
  State<ClassOverviewScreen> createState() => _ClassOverviewScreenState();
}

class _ClassOverviewScreenState extends State<ClassOverviewScreen> {
  List<ClassScoreRow> _rows = [];
  List<ClassScoreRow> _filtered = [];
  bool _loading = true;
  String? _error;

  _SortField _sortField = _SortField.score;
  bool _sortAscending = false;
  final TextEditingController _searchCtrl = TextEditingController();

  @override
  void initState() {
    super.initState();
    _load();
    _searchCtrl.addListener(_applyFilter);
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
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
      _applyFilter();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  void _applyFilter() {
    final query = _searchCtrl.text.trim().toLowerCase();
    var result = query.isEmpty
        ? List.of(_rows)
        : _rows
            .where((r) => r.studentName.toLowerCase().contains(query))
            .toList();

    result.sort((a, b) {
      final cmp = switch (_sortField) {
        _SortField.name => a.studentName.compareTo(b.studentName),
        _SortField.score => a.totalScore.compareTo(b.totalScore),
        _SortField.confidence => a.aiConfidence.compareTo(b.aiConfidence),
      };
      return _sortAscending ? cmp : -cmp;
    });

    setState(() => _filtered = result);
  }

  void _onSort(_SortField field) {
    setState(() {
      if (_sortField == field) {
        _sortAscending = !_sortAscending;
      } else {
        _sortField = field;
        _sortAscending = field == _SortField.name;
      }
    });
    _applyFilter();
  }

  // -- Build ----------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Class Scores')),
      body: _buildBody(context),
    );
  }

  Widget _buildBody(BuildContext context) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return _ErrorRetry(message: _error!, onRetry: _load);
    }
    return Column(
      children: [
        _buildSearchBar(),
        _buildSortChips(),
        Expanded(child: _buildList(context)),
      ],
    );
  }

  Widget _buildSearchBar() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
      child: TextField(
        controller: _searchCtrl,
        decoration: InputDecoration(
          hintText: 'Search by student name',
          prefixIcon: const Icon(Icons.search),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          isDense: true,
        ),
      ),
    );
  }

  Widget _buildSortChips() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      child: Row(
        children: [
          const Text('Sort: ', style: TextStyle(fontWeight: FontWeight.w500)),
          const SizedBox(width: 4),
          _sortChip('Name', _SortField.name),
          const SizedBox(width: 6),
          _sortChip('Score', _SortField.score),
          const SizedBox(width: 6),
          _sortChip('Confidence', _SortField.confidence),
        ],
      ),
    );
  }

  Widget _sortChip(String label, _SortField field) {
    final selected = _sortField == field;
    return ChoiceChip(
      label: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(label),
          if (selected)
            Icon(
              _sortAscending ? Icons.arrow_upward : Icons.arrow_downward,
              size: 14,
            ),
        ],
      ),
      selected: selected,
      onSelected: (_) => _onSort(field),
      visualDensity: VisualDensity.compact,
    );
  }

  Widget _buildList(BuildContext context) {
    if (_filtered.isEmpty) {
      return const Center(child: Text('No students found'));
    }

    return RefreshIndicator(
      onRefresh: _load,
      child: ListView.separated(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        itemCount: _filtered.length,
        separatorBuilder: (_, __) => const SizedBox(height: 8),
        itemBuilder: (context, index) {
          final row = _filtered[index];
          return _StudentTile(
            row: row,
            onTap: () => _onStudentTap(context, row),
          );
        },
      ),
    );
  }

  void _onStudentTap(BuildContext context, ClassScoreRow row) {
    context.go(
      '/teacher/exams/${widget.examId}/students/${row.studentId}',
    );
  }
}

// ---------------------------------------------------------------------------
// Student row tile
// ---------------------------------------------------------------------------

class _StudentTile extends StatelessWidget {
  final ClassScoreRow row;
  final VoidCallback onTap;

  const _StudentTile({required this.row, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isTablet = MediaQuery.sizeOf(context).shortestSide >= 600;

    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: onTap,
        child: Padding(
          padding: EdgeInsets.symmetric(
            horizontal: 16,
            vertical: isTablet ? 16 : 12,
          ),
          child: Row(
            children: [
              // Student name
              Expanded(
                flex: 3,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      row.studentName,
                      style: theme.textTheme.titleSmall,
                      overflow: TextOverflow.ellipsis,
                    ),
                    if (row.missIndicatorCount != null &&
                        row.missIndicatorCount! > 0)
                      Text(
                        '${row.missIndicatorCount} miss indicator(s)',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.error,
                        ),
                      ),
                  ],
                ),
              ),

              // Total score
              Expanded(
                flex: 2,
                child: Column(
                  children: [
                    Text(
                      row.totalScore.toStringAsFixed(1),
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text('Score', style: theme.textTheme.bodySmall),
                  ],
                ),
              ),

              // AI confidence
              Expanded(
                flex: 2,
                child: _ConfidenceBadge(confidence: row.aiConfidence),
              ),

              const Icon(Icons.chevron_right),
            ],
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Confidence badge
// ---------------------------------------------------------------------------

class _ConfidenceBadge extends StatelessWidget {
  final double confidence;

  const _ConfidenceBadge({required this.confidence});

  @override
  Widget build(BuildContext context) {
    final pct = (confidence * 100).round();
    final color = confidence >= 0.8
        ? Colors.green
        : confidence >= 0.5
            ? Colors.orange
            : Colors.red;

    return Column(
      children: [
        Text(
          '$pct%',
          style: TextStyle(
            color: color,
            fontWeight: FontWeight.w600,
          ),
        ),
        Text(
          'Confidence',
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Shared widgets
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

