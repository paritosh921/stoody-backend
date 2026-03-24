/// HTTP client with auth header injection, retry on 503, and timeout.
///
/// Wraps the `http` package and returns typed [ApiResponse] objects.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../core/auth_service.dart';
import '../core/config.dart';
import '../core/models/api_response.dart';

class NetworkService {
  NetworkService({
    required AuthService authService,
    AppConfig? config,
    http.Client? httpClient,
  })  : _auth = authService,
        _config = config ?? AppConfig.instance,
        _client = httpClient ?? http.Client();

  final AuthService _auth;
  final AppConfig _config;
  final http.Client _client;

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  Future<ApiResponse<T>> get<T>(
    String path, {
    Map<String, String>? queryParams,
    T Function(dynamic)? fromJson,
  }) async {
    final uri = _buildUri(path, queryParams);
    return _execute<T>(
      () => _client.get(uri, headers: await _headers()),
      fromJson: fromJson,
    );
  }

  Future<ApiResponse<T>> post<T>(
    String path, {
    Object? body,
    T Function(dynamic)? fromJson,
  }) async {
    final uri = _buildUri(path);
    return _execute<T>(
      () => _client.post(
        uri,
        headers: await _headers(json: true),
        body: body != null ? jsonEncode(body) : null,
      ),
      fromJson: fromJson,
    );
  }

  Future<ApiResponse<T>> patch<T>(
    String path, {
    Object? body,
    T Function(dynamic)? fromJson,
  }) async {
    final uri = _buildUri(path);
    return _execute<T>(
      () => _client.patch(
        uri,
        headers: await _headers(json: true),
        body: body != null ? jsonEncode(body) : null,
      ),
      fromJson: fromJson,
    );
  }

  void dispose() {
    _client.close();
  }

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  Uri _buildUri(String path, [Map<String, String>? query]) {
    final base = Uri.parse(_config.apiBaseUrl);
    return base.replace(
      path: '${base.path}$path',
      queryParameters: query,
    );
  }

  Future<Map<String, String>> _headers({bool json = false}) async {
    final headers = <String, String>{
      'Accept': 'application/json',
    };
    if (json) {
      headers['Content-Type'] = 'application/json; charset=utf-8';
    }
    final token = await _auth.getAccessToken();
    if (token != null) {
      headers['Authorization'] = 'Bearer $token';
    }
    return headers;
  }

  /// Execute with timeout and retry on 503.
  Future<ApiResponse<T>> _execute<T>(
    Future<http.Response> Function() request, {
    T Function(dynamic)? fromJson,
  }) async {
    int attempt = 0;
    while (true) {
      attempt++;
      try {
        final response = await request().timeout(_config.httpTimeout);
        final statusCode = response.statusCode;

        // Retry on 503 Service Unavailable with backoff.
        if (statusCode == 503 && attempt <= _config.httpMaxRetries) {
          await Future<void>.delayed(
            Duration(seconds: attempt),
          );
          continue;
        }

        return _parseResponse<T>(response, fromJson: fromJson);
      } on TimeoutException {
        if (attempt <= _config.httpMaxRetries) {
          await Future<void>.delayed(
            Duration(seconds: attempt),
          );
          continue;
        }
        return ApiResponse<T>.timeout();
      } on SocketException catch (e) {
        if (attempt <= _config.httpMaxRetries) {
          await Future<void>.delayed(
            Duration(seconds: attempt),
          );
          continue;
        }
        return ApiResponse<T>.networkError(e.message);
      }
    }
  }

  ApiResponse<T> _parseResponse<T>(
    http.Response response, {
    T Function(dynamic)? fromJson,
  }) {
    if (response.body.isEmpty) {
      return ApiResponse<T>(
        ok: response.statusCode >= 200 && response.statusCode < 300,
        statusCode: response.statusCode,
      );
    }

    try {
      final json = jsonDecode(response.body) as Map<String, dynamic>;
      return ApiResponse<T>.fromJson(
        json,
        statusCode: response.statusCode,
        fromJson: fromJson,
      );
    } on FormatException {
      return ApiResponse<T>(
        ok: false,
        statusCode: response.statusCode,
        error: 'invalid_json',
        message: 'Response body is not valid JSON',
      );
    }
  }
}
