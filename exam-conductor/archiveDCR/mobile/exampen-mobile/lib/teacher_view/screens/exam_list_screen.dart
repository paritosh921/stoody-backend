/// Teacher exam list screen.
///
/// Fetches exams from the teacher BFF and displays them in a filterable
/// list with status badges.  Tapping an exam navigates to the class
/// overview (score review) screen.
library;

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../api/teacher_api.dart';

class TeacherExamListScreen extends StatefulWidget {
  final TeacherApi api;

  const TeacherExamListScreen({super.key, required this.api});

  @override
  State<TeacherExamListScreen> createState() => _TeacherExamListScreenState();
}

class _TeacherExamListScreenState extends State<TeacherExamListScreen> {
  List<TeacherExamCard> _exams = [];
  List<TeacherExamCard> _filtered = [];
  bool _loading = true;
  String? _error;
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
      final exams = await widget.api.listExams();
      if (!mounted) return;
      setState(() {
        _exams = exams;
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
    setState(() {
      _filtered = query.isEmpty
          ? List.of(_exams)
          : _exams
              .where((e) => e.title.toLowerCase().contains(query))
              .toList();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('My Exams')),
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
          hintText: 'Search exams',
          prefixIcon: const Icon(Icons.search),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          isDense: true,
        ),
      ),
    );
  }

  Widget _buildList(BuildContext context) {
    if (_filtered.isEmpty) {
      return const Center(child: Text('No exams found'));
    }

    return RefreshIndicator(
      onRefresh: _load,
      child: ListView.separated(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        itemCount: _filtered.length,
        separatorBuilder: (_, __) => const SizedBox(height: 8),
        itemBuilder: (context, index) {
          final exam = _filtered[index];
          return _ExamTile(
            exam: exam,
            onTap: () => context.go('/teacher/exams/${exam.examId}'),
          );
        },
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Exam tile
// ---------------------------------------------------------------------------

class _ExamTile extends StatelessWidget {
  final TeacherExamCard exam;
  final VoidCallback onTap;

  const _ExamTile({required this.exam, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          child: Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      exam.title,
                      style: theme.textTheme.titleSmall,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _formatDate(exam.scheduledAt),
                      style: theme.textTheme.bodySmall,
                    ),
                    if (exam.classLabel != null) ...[
                      const SizedBox(height: 2),
                      Text(
                        exam.classLabel!,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
              _StateBadge(state: exam.state),
              const SizedBox(width: 4),
              const Icon(Icons.chevron_right),
            ],
          ),
        ),
      ),
    );
  }

  String _formatDate(DateTime dt) {
    return '${dt.day}/${dt.month}/${dt.year}';
  }
}

// ---------------------------------------------------------------------------
// State badge
// ---------------------------------------------------------------------------

class _StateBadge extends StatelessWidget {
  final String state;
  const _StateBadge({required this.state});

  @override
  Widget build(BuildContext context) {
    final (Color bg, Color fg) = switch (state) {
      'draft' => (Colors.grey.shade200, Colors.grey.shade800),
      'scheduled' => (Colors.blue.shade50, Colors.blue.shade800),
      'in_progress' => (Colors.orange.shade50, Colors.orange.shade800),
      'scoring' => (Colors.purple.shade50, Colors.purple.shade800),
      'published' => (Colors.green.shade50, Colors.green.shade800),
      _ => (Colors.grey.shade200, Colors.grey.shade800),
    };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        state.replaceAll('_', ' '),
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
