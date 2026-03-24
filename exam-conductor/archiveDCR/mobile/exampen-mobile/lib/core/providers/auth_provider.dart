/// ChangeNotifier that exposes authentication state to the widget tree.
///
/// Wraps [AuthService] and notifies listeners on login/logout so that
/// GoRouter and UI widgets react automatically.
library;

import 'package:flutter/foundation.dart';

import '../auth_service.dart';
import '../models/user.dart';
import '../storage_service.dart';

enum AuthStatus {
  /// Initial state before session restore attempt.
  unknown,

  /// User is authenticated with a valid token.
  authenticated,

  /// No valid session.
  unauthenticated,
}

class AuthProvider extends ChangeNotifier {
  AuthProvider({required StorageService storage})
      : _authService = AuthService(storage: storage);

  final AuthService _authService;

  AuthStatus _status = AuthStatus.unknown;
  AuthStatus get status => _status;

  User? get currentUser => _authService.currentUser;
  bool get isLoggedIn => _status == AuthStatus.authenticated;

  // ---------------------------------------------------------------------------
  // Bootstrap
  // ---------------------------------------------------------------------------

  /// Call once from main.dart to attempt session restore.
  Future<void> initialize() async {
    final restored = await _authService.tryRestoreSession();
    _status =
        restored ? AuthStatus.authenticated : AuthStatus.unauthenticated;
    notifyListeners();
  }

  // ---------------------------------------------------------------------------
  // Login / logout
  // ---------------------------------------------------------------------------

  Future<User> login({
    required String accessToken,
    String? refreshToken,
  }) async {
    final user = await _authService.login(
      accessToken: accessToken,
      refreshToken: refreshToken,
    );
    _status = AuthStatus.authenticated;
    notifyListeners();
    return user;
  }

  Future<void> logout() async {
    await _authService.logout();
    _status = AuthStatus.unauthenticated;
    notifyListeners();
  }

  // ---------------------------------------------------------------------------
  // Token helpers (delegated)
  // ---------------------------------------------------------------------------

  Future<String?> getAccessToken() => _authService.getAccessToken();

  Future<void> updateAccessToken(String newToken) async {
    await _authService.updateAccessToken(newToken);
    notifyListeners();
  }

  /// Expose the underlying service for [NetworkService] injection.
  AuthService get authService => _authService;
}
