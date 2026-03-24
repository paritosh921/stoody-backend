/// Environment configuration for the ExamPen mobile app.
///
/// Reads from compile-time constants injected via `--dart-define`.
/// Falls back to development defaults when not provided.
library;

enum AppEnvironment {
  development,
  staging,
  production;

  static AppEnvironment fromString(String value) {
    return AppEnvironment.values.firstWhere(
      (e) => e.name == value,
      orElse: () => AppEnvironment.development,
    );
  }
}

class AppConfig {
  final AppEnvironment environment;
  final String apiBaseUrl;
  final Duration httpTimeout;
  final int httpMaxRetries;

  const AppConfig._({
    required this.environment,
    required this.apiBaseUrl,
    required this.httpTimeout,
    required this.httpMaxRetries,
  });

  /// Singleton built from compile-time defines.
  static final AppConfig instance = _buildFromDefines();

  static AppConfig _buildFromDefines() {
    const envStr = String.fromEnvironment(
      'APP_ENV',
      defaultValue: 'development',
    );
    const apiUrl = String.fromEnvironment(
      'API_BASE_URL',
      defaultValue: 'http://10.0.2.2:8000',
    );
    const timeoutSec = int.fromEnvironment(
      'HTTP_TIMEOUT_SEC',
      defaultValue: 30,
    );
    const maxRetries = int.fromEnvironment(
      'HTTP_MAX_RETRIES',
      defaultValue: 3,
    );

    return AppConfig._(
      environment: AppEnvironment.fromString(envStr),
      apiBaseUrl: apiUrl,
      httpTimeout: Duration(seconds: timeoutSec),
      httpMaxRetries: maxRetries,
    );
  }

  bool get isDevelopment => environment == AppEnvironment.development;
  bool get isProduction => environment == AppEnvironment.production;

  @override
  String toString() =>
      'AppConfig(env=$environment, api=$apiBaseUrl)';
}
