/// Per-pen sync progress model.
///
/// Built from the status feed's `sync` section combined with the MAC list
/// to give per-pen granularity when the hub exposes it, otherwise falls back
/// to the aggregate counters.
library;

// ---------------------------------------------------------------------------
// Per-pen sync state — mirrors hub SQLite `pen_sync_status.status`
// ---------------------------------------------------------------------------

enum PenSyncState {
  pending,
  connecting,
  syncing,
  complete,
  failed,
  timeout;

  static PenSyncState fromString(String value) {
    return PenSyncState.values.firstWhere(
      (s) => s.name == value,
      orElse: () => PenSyncState.pending,
    );
  }

  bool get isTerminal =>
      this == PenSyncState.complete ||
      this == PenSyncState.failed ||
      this == PenSyncState.timeout;
}

// ---------------------------------------------------------------------------
// Per-pen sync progress
// ---------------------------------------------------------------------------

class PenSyncProgress {
  final String penMac;
  final PenSyncState state;
  final int bytesExpected;
  final int bytesReceived;
  final String? dongleMac;
  final String? errorDetail;

  const PenSyncProgress({
    required this.penMac,
    this.state = PenSyncState.pending,
    this.bytesExpected = 0,
    this.bytesReceived = 0,
    this.dongleMac,
    this.errorDetail,
  });

  double get fraction =>
      bytesExpected == 0 ? 0.0 : bytesReceived / bytesExpected;

  bool get isComplete => state == PenSyncState.complete;
  bool get isFailed =>
      state == PenSyncState.failed || state == PenSyncState.timeout;

  factory PenSyncProgress.fromJson(Map<String, dynamic> json) {
    return PenSyncProgress(
      penMac: json['pen_mac'] as String? ?? '',
      state: PenSyncState.fromString(
        json['status'] as String? ?? 'pending',
      ),
      bytesExpected: json['bytes_expected'] as int? ?? 0,
      bytesReceived: json['bytes_received'] as int? ?? 0,
      dongleMac: json['dongle_mac'] as String?,
      errorDetail: json['error_detail'] as String?,
    );
  }

  Map<String, dynamic> toJson() => {
        'pen_mac': penMac,
        'status': state.name,
        'bytes_expected': bytesExpected,
        'bytes_received': bytesReceived,
        if (dongleMac != null) 'dongle_mac': dongleMac,
        if (errorDetail != null) 'error_detail': errorDetail,
      };
}

// ---------------------------------------------------------------------------
// Upload progress (per-pen, from upload_ledger)
// ---------------------------------------------------------------------------

class PenUploadProgress {
  final String penMac;
  final int totalChunks;
  final int ackedChunks;
  final String? uploadPath;
  final bool complete;

  const PenUploadProgress({
    required this.penMac,
    this.totalChunks = 0,
    this.ackedChunks = 0,
    this.uploadPath,
    this.complete = false,
  });

  double get fraction =>
      totalChunks == 0 ? 0.0 : ackedChunks / totalChunks;

  factory PenUploadProgress.fromJson(Map<String, dynamic> json) {
    return PenUploadProgress(
      penMac: json['pen_mac'] as String? ?? '',
      totalChunks: json['total_chunks'] as int? ?? 0,
      ackedChunks: json['acked_chunks'] as int? ?? 0,
      uploadPath: json['upload_path'] as String?,
      complete: json['complete'] as bool? ?? false,
    );
  }
}
