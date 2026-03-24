/// Objection detail screen for teacher review.
///
/// Displays the full objection context (student, question, objection text)
/// and provides approve/reject actions.  The teacher can add a resolution
/// reason before submitting a verdict.
library;

import 'package:flutter/material.dart';

import '../api/teacher_api.dart';

class ObjectionDetailScreen extends StatefulWidget {
  final ObjectionInboxItem objection;
  final TeacherApi api;

  /// Optional pre-fetched student detail for context enrichment.
  final TeacherStudentDetail? studentDetail;

  const ObjectionDetailScreen({
    super.key,
    required this.objection,
    required this.api,
    this.studentDetail,
  });

  @override
  State<ObjectionDetailScreen> createState() => _ObjectionDetailScreenState();
}

class _ObjectionDetailScreenState extends State<ObjectionDetailScreen> {
  final _reasonCtrl = TextEditingController();
  bool _submitting = false;

  @override
  void dispose() {
    _reasonCtrl.dispose();
    super.dispose();
  }

  Future<void> _submitVerdict(String verdict) async {
    final reason = _reasonCtrl.text.trim();
    if (reason.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please provide a resolution reason')),
      );
      return;
    }

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('$verdict Objection?'),
        content: Text(
          'This will $verdict the objection for question '
          '${widget.objection.questionId}.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text(verdict),
          ),
        ],
      ),
    );

    if (confirmed != true || !mounted) return;

    setState(() => _submitting = true);

    try {
      final verdictValue = verdict == 'Approve' ? 'approved' : 'rejected';
      await widget.api.resolveObjection(
        widget.objection.objectionId,
        verdict: verdictValue,
        reason: reason,
      );

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Objection ${verdict.toLowerCase()}d')),
      );
      Navigator.of(context).pop(verdict);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e')),
      );
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isTablet = MediaQuery.sizeOf(context).shortestSide >= 600;
    final obj = widget.objection;

    return Scaffold(
      appBar: AppBar(title: const Text('Objection Detail')),
      body: SingleChildScrollView(
        padding: EdgeInsets.symmetric(
          horizontal: isTablet ? 48 : 16,
          vertical: 16,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Objection info card
            _InfoCard(objection: obj),
            const SizedBox(height: 16),

            // Student detail context (if available)
            if (widget.studentDetail != null) ...[
              _StudentContextCard(detail: widget.studentDetail!),
              const SizedBox(height: 16),
            ],

            // Resolution reason
            Text('Resolution Reason', style: theme.textTheme.labelLarge),
            const SizedBox(height: 8),
            TextField(
              controller: _reasonCtrl,
              maxLines: 4,
              textCapitalization: TextCapitalization.sentences,
              decoration: InputDecoration(
                hintText: 'Explain the decision',
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Action buttons
            Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: _submitting
                        ? null
                        : () => _submitVerdict('Reject'),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: theme.colorScheme.error,
                      side: BorderSide(color: theme.colorScheme.error),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    child: const Text('Reject'),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: FilledButton(
                    onPressed: _submitting
                        ? null
                        : () => _submitVerdict('Approve'),
                    style: FilledButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    child: _submitting
                        ? const SizedBox(
                            height: 20,
                            width: 20,
                            child:
                                CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Text('Approve'),
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
// Info card
// ---------------------------------------------------------------------------

class _InfoCard extends StatelessWidget {
  final ObjectionInboxItem objection;

  const _InfoCard({required this.objection});

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
            _row(theme, 'Student', objection.studentId),
            const SizedBox(height: 8),
            _row(theme, 'Question', objection.questionId),
            const SizedBox(height: 8),
            _row(theme, 'Status', objection.status),
            const SizedBox(height: 8),
            _row(
              theme,
              'Filed',
              '${objection.filedAt.day}/${objection.filedAt.month}'
                  '/${objection.filedAt.year}',
            ),
          ],
        ),
      ),
    );
  }

  Widget _row(ThemeData theme, String label, String value) {
    return Row(
      children: [
        SizedBox(
          width: 90,
          child: Text(label, style: theme.textTheme.labelMedium),
        ),
        Expanded(
          child: Text(value, style: theme.textTheme.bodyMedium),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Student context card
// ---------------------------------------------------------------------------

class _StudentContextCard extends StatelessWidget {
  final TeacherStudentDetail detail;

  const _StudentContextCard({required this.detail});

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
            Text('Student Context', style: theme.textTheme.titleSmall),
            const SizedBox(height: 8),
            Text('Name: ${detail.studentName}'),
            Text(
              'Total Score: ${detail.totalScore.toStringAsFixed(1)}',
            ),
            Text('Questions answered: ${detail.questions.length}'),
          ],
        ),
      ),
    );
  }
}
