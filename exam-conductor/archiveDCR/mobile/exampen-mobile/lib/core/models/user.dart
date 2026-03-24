/// User model derived from Stoody JWT claims.
///
/// ExamPen does not own user identity — Stoody does. This model maps
/// the JWT payload to a local representation used by the auth layer.
library;

import 'dart:convert';

class User {
  final String userId;
  final String tenantId;
  final String role;
  final String? name;
  final String? email;
  final DateTime? tokenExpiry;

  const User({
    required this.userId,
    required this.tenantId,
    required this.role,
    this.name,
    this.email,
    this.tokenExpiry,
  });

  /// Decode JWT claims from a base64url-encoded payload segment.
  factory User.fromJwtPayload(String token) {
    final parts = token.split('.');
    if (parts.length != 3) {
      throw FormatException('Invalid JWT: expected 3 parts, got ${parts.length}');
    }

    final payload = _decodeBase64Url(parts[1]);
    final claims = jsonDecode(payload) as Map<String, dynamic>;
    return User.fromClaims(claims);
  }

  factory User.fromClaims(Map<String, dynamic> claims) {
    DateTime? expiry;
    final exp = claims['exp'];
    if (exp is int) {
      expiry = DateTime.fromMillisecondsSinceEpoch(exp * 1000, isUtc: true);
    }

    return User(
      userId: claims['sub'] as String? ?? claims['user_id'] as String? ?? '',
      tenantId: claims['tenant_id'] as String? ?? '',
      role: claims['role'] as String? ?? 'student',
      name: claims['name'] as String?,
      email: claims['email'] as String?,
      tokenExpiry: expiry,
    );
  }

  bool get isExpired {
    if (tokenExpiry == null) return true;
    return DateTime.now().toUtc().isAfter(tokenExpiry!);
  }

  bool get isTeacher => role == 'tutor' || role == 'teacher';
  bool get isStudent => role == 'student';
  bool get isInvigilator => role == 'invigilator';

  Map<String, dynamic> toJson() => {
        'user_id': userId,
        'tenant_id': tenantId,
        'role': role,
        if (name != null) 'name': name,
        if (email != null) 'email': email,
        if (tokenExpiry != null)
          'token_expiry': tokenExpiry!.toIso8601String(),
      };

  static String _decodeBase64Url(String input) {
    var padded = input.replaceAll('-', '+').replaceAll('_', '/');
    switch (padded.length % 4) {
      case 2:
        padded += '==';
      case 3:
        padded += '=';
    }
    return utf8.decode(base64Decode(padded));
  }

  @override
  String toString() => 'User($userId, $role, tenant=$tenantId)';
}
