/// Secure token storage, user preferences, and offline cache helpers.
///
/// Uses flutter_secure_storage for secrets (JWT tokens) and
/// a simple in-memory map for non-sensitive prefs. A production
/// iteration would persist prefs via shared_preferences.
library;

import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class StorageService {
  StorageService({FlutterSecureStorage? secureStorage})
      : _secure = secureStorage ??
            const FlutterSecureStorage(
              aOptions: AndroidOptions(encryptedSharedPreferences: true),
            );

  final FlutterSecureStorage _secure;

  // Non-sensitive prefs held in memory for the session.
  final Map<String, String> _prefs = {};

  // ---------------------------------------------------------------------------
  // Secure token storage
  // ---------------------------------------------------------------------------

  static const _keyAccessToken = 'exampen_access_token';
  static const _keyRefreshToken = 'exampen_refresh_token';

  Future<void> saveAccessToken(String token) =>
      _secure.write(key: _keyAccessToken, value: token);

  Future<String?> readAccessToken() =>
      _secure.read(key: _keyAccessToken);

  Future<void> saveRefreshToken(String token) =>
      _secure.write(key: _keyRefreshToken, value: token);

  Future<String?> readRefreshToken() =>
      _secure.read(key: _keyRefreshToken);

  Future<void> clearTokens() async {
    await _secure.delete(key: _keyAccessToken);
    await _secure.delete(key: _keyRefreshToken);
  }

  // ---------------------------------------------------------------------------
  // User preferences (non-sensitive, in-memory for now)
  // ---------------------------------------------------------------------------

  void setPref(String key, String value) {
    _prefs[key] = value;
  }

  String? getPref(String key) => _prefs[key];

  void removePref(String key) {
    _prefs.remove(key);
  }

  // ---------------------------------------------------------------------------
  // Offline cache helpers
  // ---------------------------------------------------------------------------

  /// Store a serialised JSON blob keyed by [cacheKey].
  Future<void> cacheJson(String cacheKey, String jsonString) =>
      _secure.write(key: 'cache_$cacheKey', value: jsonString);

  /// Retrieve a previously cached JSON blob.
  Future<String?> readCachedJson(String cacheKey) =>
      _secure.read(key: 'cache_$cacheKey');

  /// Remove a single cache entry.
  Future<void> removeCachedJson(String cacheKey) =>
      _secure.delete(key: 'cache_$cacheKey');

  /// Wipe everything — tokens, prefs, and cache.
  Future<void> clearAll() async {
    _prefs.clear();
    await _secure.deleteAll();
  }
}
