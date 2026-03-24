/// GoRouter configuration with auth guard and mode-based routing.
///
/// The app has three top-level modes:
///   1. Hub control (invigilator BLE workflow)
///   2. Teacher score view
///   3. Student score view
///
/// Auth guard redirects unauthenticated users to the login screen.
/// After login, the user's role determines the initial route.
library;

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:provider/provider.dart';

import '../core/network_service.dart';
import '../core/providers/auth_provider.dart';
import '../hub_control/screens/auth_screen.dart';
import '../hub_control/screens/exam_control_screen.dart';
import '../hub_control/screens/pen_registration_screen.dart';
import '../hub_control/screens/sync_monitor_screen.dart';
import '../hub_control/screens/upload_screen.dart';
import '../student_view/api/student_api.dart';
import '../student_view/screens/answer_viewer_screen.dart';
import '../student_view/screens/chat_screen.dart' as student_chat;
import '../student_view/screens/exam_list_screen.dart';
import '../student_view/screens/objection_file_screen.dart';
import '../student_view/screens/objection_status_screen.dart';
import '../student_view/screens/performance_screen.dart';
import '../student_view/screens/question_breakdown_screen.dart';
import '../student_view/screens/score_summary_screen.dart';
import '../student_view/screens/strength_weakness_screen.dart';
import '../teacher_view/api/teacher_api.dart';
import '../teacher_view/screens/analytics_screen.dart';
import '../teacher_view/screens/class_overview_screen.dart';
import '../teacher_view/screens/exam_list_screen.dart' as teacher_exam;
import '../teacher_view/screens/leaderboard_screen.dart';
import '../teacher_view/screens/chat_screen.dart';
import '../teacher_view/screens/objection_inbox_screen.dart';
import '../teacher_view/screens/student_detail_screen.dart';

/// Build the app router, injecting [authProvider] for redirect logic.
GoRouter buildRouter(AuthProvider authProvider) {
  return GoRouter(
    initialLocation: '/hub/auth',
    refreshListenable: authProvider,
    redirect: (BuildContext context, GoRouterState state) {
      final loggedIn = authProvider.isLoggedIn;
      final onLogin = state.matchedLocation == '/login';

      if (!loggedIn && !onLogin) return '/login';
      if (loggedIn && onLogin) return _homeForRole(authProvider);
      return null;
    },
    routes: [
      GoRoute(
        path: '/login',
        builder: (context, state) => const _LoginScreen(),
      ),

      // -----------------------------------------------------------------------
      // Hub control mode (invigilator)
      // -----------------------------------------------------------------------
      GoRoute(
        path: '/hub/auth',
        builder: (context, state) => const HubAuthScreen(),
      ),
      GoRoute(
        path: '/hub/exam',
        builder: (context, state) => const ExamControlScreen(),
      ),
      GoRoute(
        path: '/hub/register',
        builder: (context, state) => const PenRegistrationScreen(),
      ),
      GoRoute(
        path: '/hub/sync',
        builder: (context, state) => const SyncMonitorScreen(),
      ),
      GoRoute(
        path: '/hub/upload',
        builder: (context, state) => const UploadScreen(),
      ),

      // -----------------------------------------------------------------------
      // Teacher mode — class scores, student detail, objections, analytics
      // -----------------------------------------------------------------------
      ShellRoute(
        builder: (context, state, child) => _ModeShell(
          title: 'Teacher',
          child: child,
        ),
        routes: [
          GoRoute(
            path: '/teacher',
            redirect: (_, __) => '/teacher/exams',
          ),
          GoRoute(
            path: '/teacher/exams',
            builder: (context, state) {
              final api = _teacherApi(context);
              return teacher_exam.TeacherExamListScreen(api: api);
            },
          ),
          GoRoute(
            path: '/teacher/exams/:examId',
            builder: (context, state) {
              final api = _teacherApi(context);
              final examId = state.pathParameters['examId']!;
              return ClassOverviewScreen(examId: examId, api: api);
            },
            routes: [
              GoRoute(
                path: 'students/:studentId',
                builder: (context, state) {
                  final api = _teacherApi(context);
                  final examId = state.pathParameters['examId']!;
                  final studentId = state.pathParameters['studentId']!;
                  return StudentDetailScreen(
                    examId: examId,
                    studentId: studentId,
                    api: api,
                  );
                },
              ),
              GoRoute(
                path: 'leaderboard',
                builder: (context, state) {
                  final api = _teacherApi(context);
                  final examId = state.pathParameters['examId']!;
                  return LeaderboardScreen(examId: examId, api: api);
                },
              ),
              GoRoute(
                path: 'analytics',
                builder: (context, state) {
                  final api = _teacherApi(context);
                  final examId = state.pathParameters['examId']!;
                  return AnalyticsScreen(examId: examId, api: api);
                },
              ),
              GoRoute(
                path: 'objections',
                builder: (context, state) {
                  final api = _teacherApi(context);
                  final examId = state.pathParameters['examId']!;
                  return ObjectionInboxScreen(
                    api: api,
                    examId: examId,
                  );
                },
              ),
            ],
          ),
          GoRoute(
            path: '/teacher/objections',
            builder: (context, state) {
              final api = _teacherApi(context);
              return ObjectionInboxScreen(api: api);
            },
          ),
          GoRoute(
            path: '/teacher/chat/:examId/:studentId',
            builder: (context, state) {
              final net = context.read<NetworkService>();
              final auth = context.read<AuthProvider>();
              return TeacherChatScreen(
                examId: state.pathParameters['examId']!,
                studentId: state.pathParameters['studentId']!,
                studentName: state.uri.queryParameters['name'] ?? 'Student',
                network: net,
                currentUserId: auth.currentUser?.userId ?? '',
              );
            },
          ),
        ],
      ),

      // -----------------------------------------------------------------------
      // Student mode — scores, breakdown, answers, objections, performance
      // -----------------------------------------------------------------------
      ShellRoute(
        builder: (context, state, child) => _ModeShell(
          title: 'Student',
          child: child,
        ),
        routes: [
          GoRoute(
            path: '/student',
            redirect: (_, __) => '/student/exams',
          ),
          GoRoute(
            path: '/student/exams',
            builder: (context, state) {
              final api = _studentApi(context);
              return StudentExamListScreen(api: api);
            },
          ),
          GoRoute(
            path: '/student/exams/:examId',
            builder: (context, state) {
              final api = _studentApi(context);
              final examId = state.pathParameters['examId']!;
              return ScoreSummaryScreen(examId: examId, api: api);
            },
            routes: [
              GoRoute(
                path: 'breakdown',
                builder: (context, state) {
                  final api = _studentApi(context);
                  final examId = state.pathParameters['examId']!;
                  return QuestionBreakdownScreen(examId: examId, api: api);
                },
              ),
              GoRoute(
                path: 'answers/:questionId',
                builder: (context, state) {
                  final api = _studentApi(context);
                  final examId = state.pathParameters['examId']!;
                  final questionId = state.pathParameters['questionId']!;
                  return AnswerViewerScreen(
                    examId: examId,
                    questionId: questionId,
                    api: api,
                  );
                },
              ),
              GoRoute(
                path: 'objection/file',
                builder: (context, state) {
                  final api = _studentApi(context);
                  final examId = state.pathParameters['examId']!;
                  return ObjectionFileScreen(examId: examId, api: api);
                },
              ),
              GoRoute(
                path: 'chat/:teacherId',
                builder: (context, state) {
                  final api = _studentApi(context);
                  final examId = state.pathParameters['examId']!;
                  final teacherId = state.pathParameters['teacherId']!;
                  final auth = context.read<AuthProvider>();
                  return student_chat.StudentChatScreen(
                    examId: examId,
                    teacherId: teacherId,
                    api: api,
                    currentUserId: auth.currentUser?.userId ?? '',
                  );
                },
              ),
            ],
          ),
          GoRoute(
            path: '/student/objections',
            builder: (context, state) {
              final api = _studentApi(context);
              return ObjectionStatusScreen(api: api);
            },
          ),
          GoRoute(
            path: '/student/performance',
            builder: (context, state) {
              final api = _studentApi(context);
              return PerformanceScreen(api: api);
            },
          ),
          GoRoute(
            path: '/student/strengths',
            builder: (context, state) {
              final api = _studentApi(context);
              return StrengthWeaknessScreen(api: api);
            },
          ),
        ],
      ),
    ],
  );
}

// ---------------------------------------------------------------------------
// API factory helpers — create per-navigation to avoid stale closures
// ---------------------------------------------------------------------------

TeacherApi _teacherApi(BuildContext context) {
  final network = context.read<NetworkService>();
  return TeacherApi(network);
}

StudentApi _studentApi(BuildContext context) {
  final network = context.read<NetworkService>();
  return StudentApi(network);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

String _homeForRole(AuthProvider auth) {
  final role = auth.currentUser?.role;
  switch (role) {
    case 'invigilator':
      return '/hub/auth';
    case 'tutor':
    case 'teacher':
      return '/teacher';
    case 'student':
      return '/student';
    default:
      return '/hub/auth';
  }
}

// ---------------------------------------------------------------------------
// Login screen — launches Stoody SSO flow
// ---------------------------------------------------------------------------

class _LoginScreen extends StatefulWidget {
  const _LoginScreen();

  @override
  State<_LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<_LoginScreen> {
  final _tokenController = TextEditingController();
  bool _loading = false;
  String? _error;

  @override
  void dispose() {
    _tokenController.dispose();
    super.dispose();
  }

  Future<void> _handleLogin() async {
    final token = _tokenController.text.trim();
    if (token.isEmpty) {
      setState(() => _error = 'Please enter a valid token');
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final auth = context.read<AuthProvider>();
      await auth.login(accessToken: token);
      // GoRouter redirect will handle navigation to the correct home.
    } catch (e) {
      if (mounted) {
        setState(() {
          _loading = false;
          _error = 'Login failed: ${e.toString()}';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 32),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  Icons.edit_note_rounded,
                  size: 64,
                  color: theme.colorScheme.primary,
                ),
                const SizedBox(height: 16),
                Text(
                  'ExamPen',
                  style: theme.textTheme.headlineMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  'Sign in with your Stoody account',
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
                const SizedBox(height: 32),
                TextField(
                  controller: _tokenController,
                  decoration: InputDecoration(
                    labelText: 'Access Token',
                    hintText: 'Paste your Stoody JWT token',
                    border: const OutlineInputBorder(),
                    errorText: _error,
                  ),
                  obscureText: true,
                  onSubmitted: (_) => _handleLogin(),
                ),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton(
                    onPressed: _loading ? null : _handleLogin,
                    child: _loading
                        ? const SizedBox(
                            height: 20,
                            width: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Text('Sign In'),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Mode shell — shared scaffold for teacher / student modes
// ---------------------------------------------------------------------------

class _ModeShell extends StatelessWidget {
  const _ModeShell({required this.title, required this.child});

  final String title;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: Text('$title View'),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            tooltip: 'Sign out',
            onPressed: () async {
              final auth = context.read<AuthProvider>();
              await auth.logout();
            },
          ),
        ],
      ),
      body: child,
      bottomNavigationBar: _buildBottomNav(context, theme),
    );
  }

  Widget? _buildBottomNav(BuildContext context, ThemeData theme) {
    final location = GoRouterState.of(context).matchedLocation;

    if (title == 'Teacher') {
      return _TeacherBottomNav(currentLocation: location);
    }
    if (title == 'Student') {
      return _StudentBottomNav(currentLocation: location);
    }
    return null;
  }
}

class _TeacherBottomNav extends StatelessWidget {
  const _TeacherBottomNav({required this.currentLocation});
  final String currentLocation;

  @override
  Widget build(BuildContext context) {
    int index = 0;
    if (currentLocation.contains('/objections')) {
      index = 1;
    }

    return NavigationBar(
      selectedIndex: index,
      onDestinationSelected: (i) {
        switch (i) {
          case 0:
            context.go('/teacher/exams');
          case 1:
            context.go('/teacher/objections');
        }
      },
      destinations: const [
        NavigationDestination(
          icon: Icon(Icons.class_outlined),
          selectedIcon: Icon(Icons.class_),
          label: 'Exams',
        ),
        NavigationDestination(
          icon: Icon(Icons.flag_outlined),
          selectedIcon: Icon(Icons.flag),
          label: 'Objections',
        ),
      ],
    );
  }
}

class _StudentBottomNav extends StatelessWidget {
  const _StudentBottomNav({required this.currentLocation});
  final String currentLocation;

  @override
  Widget build(BuildContext context) {
    int index = 0;
    if (currentLocation.contains('/objections')) {
      index = 1;
    } else if (currentLocation.contains('/performance')) {
      index = 2;
    } else if (currentLocation.contains('/strengths')) {
      index = 3;
    }

    return NavigationBar(
      selectedIndex: index,
      onDestinationSelected: (i) {
        switch (i) {
          case 0:
            context.go('/student/exams');
          case 1:
            context.go('/student/objections');
          case 2:
            context.go('/student/performance');
          case 3:
            context.go('/student/strengths');
        }
      },
      destinations: const [
        NavigationDestination(
          icon: Icon(Icons.quiz_outlined),
          selectedIcon: Icon(Icons.quiz),
          label: 'Exams',
        ),
        NavigationDestination(
          icon: Icon(Icons.flag_outlined),
          selectedIcon: Icon(Icons.flag),
          label: 'Objections',
        ),
        NavigationDestination(
          icon: Icon(Icons.trending_up),
          label: 'Performance',
        ),
        NavigationDestination(
          icon: Icon(Icons.insights),
          label: 'Insights',
        ),
      ],
    );
  }
}

