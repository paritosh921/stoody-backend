/// ExamPen mobile app entry point.
///
/// Sets up MultiProvider with core services and launches
/// MaterialApp.router with GoRouter.
library;

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'core/config.dart';
import 'core/network_service.dart';
import 'core/providers/auth_provider.dart';
import 'core/router.dart';
import 'core/storage_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Core services — created once, injected via Provider.
  final storage = StorageService();
  final authProvider = AuthProvider(storage: storage);

  // Attempt to restore a previous session before rendering.
  await authProvider.initialize();

  final networkService = NetworkService(
    authService: authProvider.authService,
    config: AppConfig.instance,
  );

  runApp(
    ExamPenApp(
      storage: storage,
      authProvider: authProvider,
      networkService: networkService,
    ),
  );
}

class ExamPenApp extends StatelessWidget {
  const ExamPenApp({
    super.key,
    required this.storage,
    required this.authProvider,
    required this.networkService,
  });

  final StorageService storage;
  final AuthProvider authProvider;
  final NetworkService networkService;

  @override
  Widget build(BuildContext context) {
    final router = buildRouter(authProvider);

    return MultiProvider(
      providers: [
        ChangeNotifierProvider<AuthProvider>.value(value: authProvider),
        Provider<StorageService>.value(value: storage),
        Provider<NetworkService>.value(value: networkService),
      ],
      child: MaterialApp.router(
        title: 'ExamPen',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          colorSchemeSeed: const Color(0xFF1565C0),
          useMaterial3: true,
          brightness: Brightness.light,
        ),
        darkTheme: ThemeData(
          colorSchemeSeed: const Color(0xFF1565C0),
          useMaterial3: true,
          brightness: Brightness.dark,
        ),
        routerConfig: router,
      ),
    );
  }
}
