/// Objection inbox screen for teachers.
///
/// Displays a filterable list of [ObjectionInboxItem] rows.  Teachers can
/// filter by status (all, pending, resolved) and tap to review individual
/// objections.
library;

import 'package:flutter/material.dart';

import '../api/teacher_api.dart';
import 'objection_detail_screen.dart';

// ---------------------------------------------------------------------------
// Status filter
// ---------------------------------------------------------------------------

enum _StatusFilter {
  all('All'),
  pending('Pending'),
  reviewing('Reviewing'),
  resolved('Resolved');

  final String label;
  const _StatusFilter(this.label);
}

// ---------------------------------------------------------------------------
// Screen
// ---------------------------------------------------------------------------

class ObjectionInboxScreen extends StatefulWidget {
  final TeacherApi api;
  final String? examId;

  const ObjectionInboxScreen({super.key, required this.api, this.examId});

  @override
  State<ObjectionInboxScreen> createState() => _ObjectionInboxScreenState();
}

class _ObjectionInboxScreenState extends State<ObjectionInboxScreen> {
  List<ObjectionInboxItem> _items = [];
  bool _loading = true;
  String? _error;
  _StatusFilter _filter = _StatusFilter.all;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    if (widget.examId == null) {
      setState(() {
        _loading = false;
        _error = null;
      });
      return;
    }
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final items = await widget.api.listObjections(widget.examId!);
      if (!mounted) return;
      setState(() {
        _items = items;
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

  List<ObjectionInboxItem> get _filtered {
    if (_filter == _StatusFilter.all) return _items;
    return _items.where((item) {
      return switch (_filter) {
        _StatusFilter.pending =>
          item.status == 'filed' || item.status == 'assigned',
        _StatusFilter.reviewing => item.status == 'reviewing',
        _StatusFilter.resolved => item.status == 'resolved',
        _StatusFilter.all => true,
      };
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Objection Inbox')),
      body: _buildBody(context),
    );
  }

  Widget _buildBody(BuildContext context) {
    if (widget.examId == null) {
      return const Center(
        child: Text('Select an exam to view its objections.'),
      );
    }
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return _ErrorRetry(message: _error!, onRetry: _load);
    }
    return Column(
      children: [
        _buildFilterBar(),
        Expanded(child: _buildList(context)),
      ],
    );
  }

  Widget _buildFilterBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(
          children: _StatusFilter.values.map((f) {
            final count = f == _StatusFilter.all
                ? _items.length
                : _items.where((item) {
                    return switch (f) {
                      _StatusFilter.pending =>
                        item.status == 'filed' || item.status == 'assigned',
                      _StatusFilter.reviewing => item.status == 'reviewing',
                      _StatusFilter.resolved => item.status == 'resolved',
                      _StatusFilter.all => true,
                    };
                  }).length;

            return Padding(
              padding: const EdgeInsets.only(right: 8),
              child: FilterChip(
                label: Text('${f.label} ($count)'),
                selected: _filter == f,
                onSelected: (_) => setState(() => _filter = f),
              ),
            );
          }).toList(),
        ),
      ),
    );
  }

  Widget _buildList(BuildContext context) {
    final items = _filtered;
    if (items.isEmpty) {
      return const Center(child: Text('No objections match the filter'));
    }

    return RefreshIndicator(
      onRefresh: _load,
      child: ListView.separated(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
        itemCount: items.length,
        separatorBuilder: (_, __) => const SizedBox(height: 8),
        itemBuilder: (_, index) {
          final item = items[index];
          return _ObjectionTile(
            item: item,
            onTap: () => _onObjectionTap(context, item),
          );
        },
      ),
    );
  }

  void _onObjectionTap(BuildContext context, ObjectionInboxItem item) {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ObjectionDetailScreen(
          api: widget.api,
          objectionId: item.objectionId,
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Objection tile
// ---------------------------------------------------------------------------

class _ObjectionTile extends StatelessWidget {
  final ObjectionInboxItem item;
  final VoidCallback onTap;

  const _ObjectionTile({required this.item, required this.onTap});

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
          padding: EdgeInsets.all(isTablet ? 20 : 14),
          child: Row(
            children: [
              _StatusDot(status: item.status),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Student: ${item.studentId}',
                      style: theme.textTheme.titleSmall,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 2),
                    Text(
                      'Question: ${item.questionId}',
                      style: theme.textTheme.bodySmall,
                    ),
                  ],
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  _StatusChip(status: item.status),
                  const SizedBox(height: 4),
                  Text(
                    _formatDate(item.filedAt),
                    style: theme.textTheme.bodySmall,
                  ),
                ],
              ),
              const SizedBox(width: 4),
              const Icon(Icons.chevron_right, size: 20),
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
// Status widgets
// ---------------------------------------------------------------------------

class _StatusDot extends StatelessWidget {
  final String status;
  const _StatusDot({required this.status});

  Color get _color => switch (status) {
        'filed' || 'assigned' => Colors.orange,
        'reviewing' => Colors.blue,
        'resolved' => Colors.green,
        _ => Colors.grey,
      };

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 10,
      height: 10,
      decoration: BoxDecoration(color: _color, shape: BoxShape.circle),
    );
  }
}

class _StatusChip extends StatelessWidget {
  final String status;
  const _StatusChip({required this.status});

  @override
  Widget build(BuildContext context) {
    final (Color bg, Color fg) = switch (status) {
      'filed' || 'assigned' => (Colors.orange.shade50, Colors.orange.shade800),
      'reviewing' => (Colors.blue.shade50, Colors.blue.shade800),
      'resolved' => (Colors.green.shade50, Colors.green.shade800),
      _ => (Colors.grey.shade200, Colors.grey.shade800),
    };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        status,
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
