/// Answer image viewer with pinch-zoom.
///
/// Fetches [AnswerInsight] for a single question and displays the answer
/// image with interactive zoom and pan.  Below the image, shows recognized
/// text, step breakdown, and AI feedback.
library;

import 'package:flutter/material.dart';

import '../api/student_api.dart';

class AnswerViewerScreen extends StatefulWidget {
  final String examId;
  final String questionId;
  final StudentApi api;

  const AnswerViewerScreen({
    super.key,
    required this.examId,
    required this.questionId,
    required this.api,
  });

  @override
  State<AnswerViewerScreen> createState() => _AnswerViewerScreenState();
}

class _AnswerViewerScreenState extends State<AnswerViewerScreen> {
  AnswerInsight? _insight;
  bool _loading = true;
  String? _error;
  final _transformCtrl = TransformationController();

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _transformCtrl.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final insight = await widget.api.getAnswerInsight(
        widget.examId,
        widget.questionId,
      );
      if (!mounted) return;
      setState(() {
        _insight = insight;
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
        title: Text('Q${widget.questionId} Answer'),
        actions: [
          if (_insight != null)
            IconButton(
              icon: const Icon(Icons.zoom_out_map),
              tooltip: 'Reset zoom',
              onPressed: () => _transformCtrl.value = Matrix4.identity(),
            ),
        ],
      ),
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

    final insight = _insight!;
    final isTablet = MediaQuery.sizeOf(context).shortestSide >= 600;

    return isTablet ? _buildTabletLayout(insight) : _buildPhoneLayout(insight);
  }

  // -- Phone layout: image on top, details below ----------------------------

  Widget _buildPhoneLayout(AnswerInsight insight) {
    return ListView(
      children: [
        SizedBox(
          height: 320,
          child: _buildZoomableImage(insight),
        ),
        Padding(
          padding: const EdgeInsets.all(16),
          child: _InsightDetails(insight: insight),
        ),
      ],
    );
  }

  // -- Tablet layout: image left, details right -----------------------------

  Widget _buildTabletLayout(AnswerInsight insight) {
    return Row(
      children: [
        Expanded(
          flex: 3,
          child: _buildZoomableImage(insight),
        ),
        Expanded(
          flex: 2,
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: _InsightDetails(insight: insight),
          ),
        ),
      ],
    );
  }

  Widget _buildZoomableImage(AnswerInsight insight) {
    return InteractiveViewer(
      transformationController: _transformCtrl,
      minScale: 0.5,
      maxScale: 5.0,
      child: Image.network(
        insight.answerImageUri,
        fit: BoxFit.contain,
        loadingBuilder: (_, child, progress) {
          if (progress == null) return child;
          return const Center(child: CircularProgressIndicator());
        },
        errorBuilder: (_, __, ___) {
          return const Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.broken_image, size: 48, color: Colors.grey),
                SizedBox(height: 8),
                Text('Failed to load image'),
              ],
            ),
          );
        },
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Insight details
// ---------------------------------------------------------------------------

class _InsightDetails extends StatelessWidget {
  final AnswerInsight insight;

  const _InsightDetails({required this.insight});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final confPct = (insight.confidence * 100).round();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Confidence badge
        Row(
          children: [
            Text('AI Confidence', style: theme.textTheme.labelLarge),
            const Spacer(),
            _ConfidenceChip(pct: confPct, confidence: insight.confidence),
          ],
        ),
        const Divider(height: 24),

        // Recognized text
        Text('Recognized Text', style: theme.textTheme.labelLarge),
        const SizedBox(height: 8),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: theme.colorScheme.surfaceContainerHighest,
            borderRadius: BorderRadius.circular(8),
          ),
          child: SelectableText(
            insight.recognizedText.isNotEmpty
                ? insight.recognizedText
                : '(No text recognized)',
            style: theme.textTheme.bodyMedium,
          ),
        ),

        // Step breakdown
        if (insight.stepBreakdown.isNotEmpty) ...[
          const SizedBox(height: 20),
          Text('Step Breakdown', style: theme.textTheme.labelLarge),
          const SizedBox(height: 8),
          ...insight.stepBreakdown.asMap().entries.map((entry) {
            return Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${entry.key + 1}. ',
                    style: theme.textTheme.bodyMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
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

        // Feedback
        if (insight.feedback != null && insight.feedback!.isNotEmpty) ...[
          const SizedBox(height: 20),
          Text('AI Feedback', style: theme.textTheme.labelLarge),
          const SizedBox(height: 8),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: theme.colorScheme.tertiaryContainer.withValues(alpha: 0.3),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(
                color: theme.colorScheme.tertiary.withValues(alpha: 0.3),
              ),
            ),
            child: Text(
              insight.feedback!,
              style: theme.textTheme.bodyMedium,
            ),
          ),
        ],
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Confidence chip
// ---------------------------------------------------------------------------

class _ConfidenceChip extends StatelessWidget {
  final int pct;
  final double confidence;

  const _ConfidenceChip({required this.pct, required this.confidence});

  @override
  Widget build(BuildContext context) {
    final color = confidence >= 0.8
        ? Colors.green
        : confidence >= 0.5
            ? Colors.orange
            : Colors.red;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        '$pct%',
        style: TextStyle(
          color: color,
          fontWeight: FontWeight.bold,
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
