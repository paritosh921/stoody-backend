/// Per-question score breakdown screen.
///
/// Lists every question in the exam with marks obtained vs max marks,
/// AI confidence, and miss indicators.  Tapping a question navigates
/// to the answer viewer.
library;

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../api/student_api.dart';

class QuestionBreakdownScreen extends StatefulWidget {
  final String examId;
  final StudentApi api;

  const QuestionBreakdownScreen({
    super.key,
    required this.examId,
    required this.api,
  });

  @override
  State<QuestionBreakdownScreen> createState() =>
      _QuestionBreakdownScreenState();
}

class _QuestionBreakdownScreenState extends State<QuestionBreakdownScreen> {
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
      appBar: AppBar(title: const Text('Question Breakdown')),
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

    final questions = _score!.questions;
    if (questions.isEmpty) {
      return const Center(child: Text('No questions available'));
    }

    return RefreshIndicator(
      onRefresh: _load,
      child: ListView.separated(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        itemCount: questions.length,
        separatorBuilder: (_, __) => const SizedBox(height: 8),
        itemBuilder: (context, index) {
          final q = questions[index];
          return _QuestionTile(
            question: q,
            index: index + 1,
            onTap: () => _onQuestionTap(q),
          );
        },
      ),
    );
  }

  void _onQuestionTap(StudentQuestionScore question) {
    context.go(
      '/student/exams/${widget.examId}/answers/${question.questionId}',
    );
  }
}

// ---------------------------------------------------------------------------
// Question tile
// ---------------------------------------------------------------------------

class _QuestionTile extends StatelessWidget {
  final StudentQuestionScore question;
  final int index;
  final VoidCallback onTap;

  const _QuestionTile({
    required this.question,
    required this.index,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isTablet = MediaQuery.sizeOf(context).shortestSide >= 600;
    final fraction = question.maxMarks > 0
        ? question.marksObtained / question.maxMarks
        : 0.0;
    final barColor = fraction >= 0.7
        ? Colors.green
        : fraction >= 0.4
            ? Colors.orange
            : Colors.red;

    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: onTap,
        child: Padding(
          padding: EdgeInsets.all(isTablet ? 20 : 14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header row
              Row(
                children: [
                  CircleAvatar(
                    radius: 16,
                    backgroundColor: theme.colorScheme.primaryContainer,
                    child: Text(
                      'Q$index',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                        color: theme.colorScheme.onPrimaryContainer,
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'Question ${question.questionId}',
                      style: theme.textTheme.titleSmall,
                    ),
                  ),
                  Text(
                    '${question.marksObtained.toStringAsFixed(1)} / '
                    '${question.maxMarks.toStringAsFixed(1)}',
                    style: theme.textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(width: 4),
                  const Icon(Icons.chevron_right, size: 20),
                ],
              ),
              const SizedBox(height: 10),

              // Score bar
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: fraction.clamp(0.0, 1.0),
                  minHeight: 6,
                  backgroundColor: barColor.withValues(alpha: 0.15),
                  valueColor: AlwaysStoppedAnimation(barColor),
                ),
              ),
              const SizedBox(height: 8),

              // Meta row
              Row(
                children: [
                  if (question.aiConfidence != null) ...[
                    _ConfidenceChip(confidence: question.aiConfidence!),
                    const SizedBox(width: 8),
                  ],
                  if (question.missIndicator != null)
                    _MissChip(indicator: question.missIndicator!),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Chips
// ---------------------------------------------------------------------------

class _ConfidenceChip extends StatelessWidget {
  final double confidence;
  const _ConfidenceChip({required this.confidence});

  @override
  Widget build(BuildContext context) {
    final pct = (confidence * 100).round();
    final color = confidence >= 0.8
        ? Colors.green
        : confidence >= 0.5
            ? Colors.orange
            : Colors.red;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        'AI $pct%',
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w600,
          color: color,
        ),
      ),
    );
  }
}

class _MissChip extends StatelessWidget {
  final String indicator;
  const _MissChip({required this.indicator});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: Colors.red.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        indicator,
        style: const TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w600,
          color: Colors.red,
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
