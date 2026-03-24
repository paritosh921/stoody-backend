/// Generic API response wrapper.
///
/// All HTTP responses from the ExamPen backend are expected to follow
/// the `{ ok, data?, error?, message? }` envelope. This wrapper
/// provides typed access with null-safe parsing.
library;

class ApiResponse<T> {
  final bool ok;
  final T? data;
  final String? error;
  final String? message;
  final int statusCode;

  const ApiResponse({
    required this.ok,
    this.data,
    this.error,
    this.message,
    required this.statusCode,
  });

  /// Parse a JSON map using the provided [fromJson] to hydrate [data].
  factory ApiResponse.fromJson(
    Map<String, dynamic> json, {
    required int statusCode,
    T Function(dynamic json)? fromJson,
  }) {
    T? parsed;
    if (fromJson != null && json.containsKey('data') && json['data'] != null) {
      parsed = fromJson(json['data']);
    }
    return ApiResponse<T>(
      ok: json['ok'] as bool? ?? (statusCode >= 200 && statusCode < 300),
      data: parsed,
      error: json['error'] as String?,
      message: json['message'] as String?,
      statusCode: statusCode,
    );
  }

  /// Convenience for network / transport failures.
  factory ApiResponse.networkError(String message) {
    return ApiResponse<T>(
      ok: false,
      error: 'network_error',
      message: message,
      statusCode: 0,
    );
  }

  /// Convenience for timeout failures.
  factory ApiResponse.timeout() {
    return ApiResponse<T>(
      ok: false,
      error: 'timeout',
      message: 'Request timed out',
      statusCode: 0,
    );
  }

  bool get isUnauthorized => statusCode == 401;
  bool get isServiceUnavailable => statusCode == 503;

  @override
  String toString() =>
      'ApiResponse(ok=$ok, status=$statusCode, error=$error)';
}
