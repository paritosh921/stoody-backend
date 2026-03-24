/// Inline score edit screen.
///
/// Allows a teacher to override the AI-assigned score for a specific question.
/// The teacher must provide a reason and can optionally mark individual steps.
/// Submits a [TeacherScoreOverrideRequest] via the teacher BFF PATCH endpoint.
library;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../api/teacher_api.dart';

class ScoreEditScreen extends StatefulWidget {
  final String examId;
  final String studentId;
  final QuestionDetail question;
  final TeacherApi api;

  const ScoreEditScreen({
    super.key,
    required this.examId,
    required this.studentId,
    required this.question,
    required this.api,
  });

  @override
  State<ScoreEditScreen> createState() => _ScoreEditScreenState();
}

class _ScoreEditScreenState extends State<ScoreEditScreen> {
  final _formKey = GlobalKey<FormState>();
  late final TextEditingController _scoreCtrl;
  late final TextEditingController _reasonCtrl;
  bool _submitting = false;

  @override
  void initState() {
    super.initState();
    _scoreCtrl = TextEditingController(
      text: widget.question.currentScore.toStringAsFixed(1),
    );
    _reasonCtrl = TextEditingController();
  }

  @override
  void dispose() {
    _scoreCtrl.dispose();
    _reasonCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;

    final newScore = double.tryParse(_scoreCtrl.text.trim());
    if (newScore == null) return;

    setState(() => _submitting = true);

    try {
      final result = await widget.api.overrideScore(
        widget.examId,
        widget.studentId,
        TeacherScoreOverrideRequest(
          questionId: widget.question.questionId,
          newScore: newScore,
          reason: _reasonCtrl.text.trim(),
        ),
      );
      if (!mounted) return;

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Score updated to ${newScore.toStringAsFixed(1)}',
          ),
        ),
      );
      Navigator.of(context).pop(result);
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

    return Scaffold(
      appBar: AppBar(title: const Text('Edit Score')),
      body: SingleChildScrollView(
        padding: EdgeInsets.symmetric(
          horizontal: isTablet ? 48 : 16,
          vertical: 16,
        ),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Current score card
              _CurrentScoreCard(question: widget.question),
              const SizedBox(height: 24),

              // Recognized text preview
              if (widget.question.recognizedText != null &&
                  widget.question.recognizedText!.isNotEmpty) ...[
                Text(
                  'Recognized Text',
                  style: theme.textTheme.labelLarge,
                ),
                const SizedBox(height: 8),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: theme.colorScheme.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(widget.question.recognizedText!),
                ),
                const SizedBox(height: 24),
              ],

              // New score input
              Text('New Score', style: theme.textTheme.labelLarge),
              const SizedBox(height: 8),
              TextFormField(
                controller: _scoreCtrl,
                keyboardType:
                    const TextInputType.numberWithOptions(decimal: true),
                inputFormatters: [
                  FilteringTextInputFormatter.allow(RegExp(r'[\d.]')),
                ],
                decoration: InputDecoration(
                  hintText: 'Enter new score',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                validator: (value) {
                  if (value == null || value.trim().isEmpty) {
                    return 'Score is required';
                  }
                  final parsed = double.tryParse(value.trim());
                  if (parsed == null || parsed < 0) {
                    return 'Enter a valid non-negative number';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 20),

              // Reason input
              Text('Reason for Override', style: theme.textTheme.labelLarge),
              const SizedBox(height: 8),
              TextFormField(
                controller: _reasonCtrl,
                maxLines: 4,
                textCapitalization: TextCapitalization.sentences,
                decoration: InputDecoration(
                  hintText: 'Explain why the score is being changed',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                validator: (value) {
                  if (value == null || value.trim().isEmpty) {
                    return 'Reason is required';
                  }
                  if (value.trim().length < 10) {
                    return 'Please provide a more detailed reason';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 32),

              // Submit button
              FilledButton(
                onPressed: _submitting ? null : _submit,
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: _submitting
                    ? const SizedBox(
                        height: 20,
                        width: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text('Submit Override'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Current score card
// ---------------------------------------------------------------------------

class _CurrentScoreCard extends StatelessWidget {
  final QuestionDetail question;

  const _CurrentScoreCard({required this.question});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final confPct = (question.confidence * 100).round();

    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            CircleAvatar(
              radius: 24,
              backgroundColor: theme.colorScheme.primaryContainer,
              child: Text(
                question.questionId,
                style: TextStyle(
                  color: theme.colorScheme.onPrimaryContainer,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Current Score: ${question.currentScore.toStringAsFixed(1)}',
                    style: theme.textTheme.titleMedium,
                  ),
                  Text(
                    'AI Confidence: $confPct%',
                    style: theme.textTheme.bodySmall,
                  ),
                  if (question.missIndicator != null)
                    Text(
                      'Miss: ${question.missIndicator}',
                      style: TextStyle(
                        color: theme.colorScheme.error,
                        fontSize: 12,
                      ),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
