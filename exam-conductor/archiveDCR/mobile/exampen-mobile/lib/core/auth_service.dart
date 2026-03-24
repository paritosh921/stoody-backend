/// Authentication service — token lifecycle, JWT decode, expiry check.
///
/// Delegates secure persistence to [StorageService] and exposes a
/// synchronous [currentUser] for the rest of the app.
library;

import 'dart:async';

import '../core/models/user.dart';
import '../core/storage_service.dart';

class AuthService {
  AuthService({required StorageService storage}) : _storage = storage;

  final StorageService _storage;

  User? _currentUser;
  User? get currentUser => _currentUser;

  bool get isLoggedIn => _currentUser != null && !_currentUser!.isExpired;

  // ---------------------------------------------------------------------------
  // Bootstrap — call once at app start
  // ---------------------------------------------------------------------------

  /// Attempt to restore a previous session from secure storage.
  Future<bool> tryRestoreSession() async {
    final token = await _storage.readAccessToken();
    if (token == null) return false;

    try {
      final user = User.fromJwtPayload(token);
      if (user.isExpired) {
        await _storage.clearTokens();
        return false;
      }
      _currentUser = user;
      return true;
    } on FormatException {
      await _storage.clearTokens();
      return false;
    }
  }

  // ---------------------------------------------------------------------------
  // Login / logout
  // ---------------------------------------------------------------------------

  /// Persist tokens returned by the backend and hydrate [currentUser].
  Future<User> login({
    required String accessToken,
    String? refreshToken,
  }) async {
    final user = User.fromJwtPayload(accessToken);
    await _storage.saveAccessToken(accessToken);
    if (refreshToken != null) {
      await _storage.saveRefreshToken(refreshToken);
    }
    _currentUser = user;
    return user;
  }

  Future<void> logout() async {
    _currentUser = null;
    await _storage.clearTokens();
  }

  // ---------------------------------------------------------------------------
  // Token access (for HTTP injection)
  // ---------------------------------------------------------------------------

  /// Returns the raw access token or `null` if not logged in.
  Future<String?> getAccessToken() => _storage.readAccessToken();

  /// Returns the raw refresh token or `null`.
  Future<String?> getRefreshToken() => _storage.readRefreshToken();

  /// Replace the access token after a refresh-token exchange.
  Future<User> updateAccessToken(String newToken) async {
    final user = User.fromJwtPayload(newToken);
    await _storage.saveAccessToken(newToken);
    _currentUser = user;
    return user;
  }

  // ---------------------------------------------------------------------------
  // Expiry helpers
  // ---------------------------------------------------------------------------

  /// Duration until the current token expires, or `null` if unknown.
  Duration? get timeUntilExpiry {
    final exp = _currentUser?.tokenExpiry;
    if (exp == null) return null;
    final remaining = exp.difference(DateTime.now().toUtc());
    return remaining.isNegative ? Duration.zero : remaining;
  }
}
