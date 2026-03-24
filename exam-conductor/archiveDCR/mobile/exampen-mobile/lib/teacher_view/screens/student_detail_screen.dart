/// Per-student drill-down screen.
///
/// Shows a per-question breakdown with AI analysis including recognized text,
/// confidence scores, and miss indicators.  Each question row can expand to
/// show recognized text and a link to the copy image.
library;

import 'package:flutter/material.dart';

import '../api/teacher_api.dart';

class StudentDetailScreen extends StatefulWidget {
  final String examId;
  final String studentId;
  final TeacherApi api;

  const StudentDetailScreen({
    super.key,
    required this.examId,
    required this.studentId,
    required this.api,
  });

  @override
  State<StudentDetailScreen> createState() => _StudentDetailScreenState();
}

class _StudentDetailScreenState extends State<StudentDetailScreen> {
  TeacherStudentDetail? _detail;
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
      final detail = await widget.api.getStudentDetail(
        widget.examId,
        widget.studentId,
      );
      if (!mounted) return;
      setState(() {
        _detail = detail;
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
      appBar: AppBar(
        title: Text(_detail?.studentName ?? 'Student Detail'),
      ),
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
    final detail = _detail!;
    return RefreshIndicator(
      onRefresh: _load,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _SummaryHeader(detail: detail),
          const SizedBox(height: 16),
          Text(
            'Per-Question Breakdown',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 8),
          ...detail.questions.map(
            (q) => _QuestionTile(
              question: q,
              examId: widget.examId,
              studentId: widget.studentId,
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Summary header
// ---------------------------------------------------------------------------

class _SummaryHeader extends StatelessWidget {
  final TeacherStudentDetail detail;

  const _SummaryHeader({required this.detail});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isTablet = MediaQuery.sizeOf(context).shortestSide >= 600;

    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: EdgeInsets.all(isTablet ? 24 : 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              detail.studentName,
              style: theme.textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                _StatBox(
                  label: 'Total Score',
                  value: detail.totalScore.toStringAsFixed(1),
                ),
                const SizedBox(width: 16),
                _StatBox(
                  label: 'Questions',
                  value: '${detail.questions.length}',
                ),
              ],
            ),
            if (detail.answerPages.isNotEmpty) ...[
              const SizedBox(height: 12),
              Text(
                '${detail.answerPages.length} answer page(s) available',
                style: theme.textTheme.bodySmall,
              ),
            ],
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Stat box
// ---------------------------------------------------------------------------

class _StatBox extends StatelessWidget {
  final String label;
  final String value;

  const _StatBox({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(value, style: theme.textTheme.headlineMedium),
        Text(label, style: theme.textTheme.bodySmall),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Expandable question tile
// ---------------------------------------------------------------------------

class _QuestionTile extends StatelessWidget {
  final QuestionDetail question;
  final String examId;
  final String studentId;

  const _QuestionTile({
    required this.question,
    required this.examId,
    required this.studentId,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final confPct = (question.confidence * 100).round();
    final confColor = question.confidence >= 0.8
        ? Colors.green
        : question.confidence >= 0.5
            ? Colors.orange
            : Colors.red;

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: ExpansionTile(
        leading: CircleAvatar(
          backgroundColor: theme.colorScheme.primaryContainer,
          child: Text(
            question.questionId,
            style: TextStyle(
              fontSize: 12,
              color: theme.colorScheme.onPrimaryContainer,
            ),
          ),
        ),
        title: Row(
          children: [
            Text(
              'Score: ${question.currentScore.toStringAsFixed(1)}',
              style: theme.textTheme.titleSmall,
            ),
            const Spacer(),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
              decoration: BoxDecoration(
                color: confColor.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                '$confPct%',
                style: TextStyle(
                  color: confColor,
                  fontWeight: FontWeight.w600,
                  fontSize: 12,
                ),
              ),
            ),
          ],
        ),
        subtitle: question.missIndicator != null
            ? Text(
                'Miss: ${question.missIndicator}',
                style: TextStyle(
                  color: theme.colorScheme.error,
                  fontSize: 12,
                ),
              )
            : null,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (question.recognizedText != null &&
                    question.recognizedText!.isNotEmpty) ...[
                  Text(
                    'Recognized Text',
                    style: theme.textTheme.labelMedium,
                  ),
                  const SizedBox(height: 4),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: theme.colorScheme.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      question.recognizedText!,
                      style: theme.textTheme.bodyMedium,
                    ),
                  ),
                  const SizedBox(height: 12),
                ],
                if (question.copyImageUri != null)
                  OutlinedButton.icon(
                    onPressed: () {
                      // Navigate to image viewer
                    },
                    icon: const Icon(Icons.image, size: 18),
                    label: const Text('View Answer Image'),
                  ),
              ],
            ),
          ),
        ],
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
