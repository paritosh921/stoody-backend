/// Objection filing screen.
///
/// Allows a student to file an objection against a specific question.
/// The student selects a question from their exam, enters objection text,
/// and submits via the student BFF POST endpoint.
library;

import 'package:flutter/material.dart';

import '../api/student_api.dart';

class ObjectionFileScreen extends StatefulWidget {
  final String examId;
  final StudentApi api;

  /// Pre-populated question list from the score view.
  final List<StudentQuestionScore> questions;

  const ObjectionFileScreen({
    super.key,
    required this.examId,
    required this.api,
    this.questions = const [],
  });

  @override
  State<ObjectionFileScreen> createState() => _ObjectionFileScreenState();
}

class _ObjectionFileScreenState extends State<ObjectionFileScreen> {
  final _formKey = GlobalKey<FormState>();
  final _textCtrl = TextEditingController();
  String? _selectedQuestionId;
  bool _submitting = false;

  @override
  void dispose() {
    _textCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    if (_selectedQuestionId == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select a question')),
      );
      return;
    }

    setState(() => _submitting = true);

    try {
      final objection = await widget.api.fileObjection(
        CreateObjectionRequest(
          examId: widget.examId,
          questionId: _selectedQuestionId!,
          objectionText: _textCtrl.text.trim(),
        ),
      );
      if (!mounted) return;

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Objection filed successfully')),
      );
      Navigator.of(context).pop(objection);
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
      appBar: AppBar(title: const Text('File Objection')),
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
              // Info banner
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: theme.colorScheme.secondaryContainer
                      .withValues(alpha: 0.3),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  children: [
                    Icon(
                      Icons.info_outline,
                      color: theme.colorScheme.secondary,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Select the question you want to object to and '
                        'provide a clear explanation.',
                        style: theme.textTheme.bodySmall,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),

              // Question selector
              Text('Select Question', style: theme.textTheme.labelLarge),
              const SizedBox(height: 8),
              _buildQuestionSelector(theme),
              const SizedBox(height: 24),

              // Objection text
              Text('Objection Details', style: theme.textTheme.labelLarge),
              const SizedBox(height: 8),
              TextFormField(
                controller: _textCtrl,
                maxLines: 6,
                textCapitalization: TextCapitalization.sentences,
                decoration: InputDecoration(
                  hintText:
                      'Explain why you believe the score should be changed...',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                validator: (value) {
                  if (value == null || value.trim().isEmpty) {
                    return 'Objection text is required';
                  }
                  if (value.trim().length < 20) {
                    return 'Please provide a more detailed explanation '
                        '(at least 20 characters)';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 32),

              // Submit
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
                    : const Text('Submit Objection'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildQuestionSelector(ThemeData theme) {
    if (widget.questions.isEmpty) {
      // Fallback: free-text question ID input
      return TextFormField(
        decoration: InputDecoration(
          hintText: 'Enter question ID',
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        onChanged: (value) => _selectedQuestionId = value.trim(),
        validator: (value) {
          if (value == null || value.trim().isEmpty) {
            return 'Question ID is required';
          }
          return null;
        },
      );
    }

    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: widget.questions.map((q) {
        final selected = _selectedQuestionId == q.questionId;
        return ChoiceChip(
          label: Text(
            'Q${q.questionId} '
            '(${q.marksObtained.toStringAsFixed(1)}/'
            '${q.maxMarks.toStringAsFixed(1)})',
          ),
          selected: selected,
          onSelected: (_) {
            setState(() => _selectedQuestionId = q.questionId);
          },
        );
      }).toList(),
    );
  }
}
