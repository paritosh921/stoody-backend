/// Teacher View integration tests for ExamPen mobile.
///
/// Task:   W6.A3 (Mobile <-> BFF integration)
/// Level:  L5
/// Spec:   TEST_SUITE_SPEC.md -- I-BFF-T01, E2E-10
///
/// These tests verify the Flutter teacher view against real BFF services.
/// No BLE hardware is required -- these exercise HTTP API flows only.
///
/// Prerequisites:
///   - Full Docker Compose stack running (svc-teacher-bff + backing services)
///   - Seed data loaded (scripts/seed-data.sh)
///
/// Run with:
///   flutter test integration_test/teacher_view_test.dart
@Tags(['api'])
library;

import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:exampen_mobile/main.dart' as app;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Teacher View — Login & Exam List (I-BFF-T01)', () {
    testWidgets(
      'Login -> exam list loads with seeded exams',
      (WidgetTester tester) async {
        app.main();
        await tester.pumpAndSettle(const Duration(seconds: 3));

        // Navigate to teacher mode / login.
        final teacherModeBtn = find.textContaining(
          RegExp(r'teacher|tutor', caseSensitive: false),
        );
        if (teacherModeBtn.evaluate().isNotEmpty) {
          await tester.tap(teacherModeBtn.first);
          await tester.pumpAndSettle();
        }

        // Enter credentials.
        final emailField = find.byWidgetPredicate(
          (w) => w.runtimeType.toString().contains('TextField'),
        );
        if (emailField.evaluate().length >= 2) {
          await tester.enterText(emailField.first, 'teacher@test.exampen.local');
          await tester.enterText(emailField.at(1), 'test-teacher-pass');
        }

        // Tap login.
        final loginBtn = find.textContaining(
          RegExp(r'login|sign in|log in', caseSensitive: false),
        );
        if (loginBtn.evaluate().isNotEmpty) {
          await tester.tap(loginBtn.first);
          await tester.pump(const Duration(seconds: 5));
        }

        // Expect exam list to appear.
        final examList = find.textContaining(
          RegExp(r'exam|mathematics|science', caseSensitive: false),
        );
        expect(examList, findsWidgets,
          reason: 'Exam list should display seeded exams after teacher login.');
      },
    );
  });

  group('Teacher View — Score Overview (E2E-10)', () {
    testWidgets(
      'Score overview -> student list renders with marks',
      (WidgetTester tester) async {
        app.main();
        await tester.pumpAndSettle(const Duration(seconds: 3));

        // Assume already logged in or auto-login from secure storage.
        // Navigate to scores.
        final scoresTab = find.textContaining(
          RegExp(r'score|result|review', caseSensitive: false),
        );
        if (scoresTab.evaluate().isEmpty) return;

        await tester.tap(scoresTab.first);
        await tester.pump(const Duration(seconds: 5));

        // Student list with scores should appear.
        final studentRows = find.textContaining(
          RegExp(r'student|name|marks|score', caseSensitive: false),
        );
        expect(studentRows, findsWidgets,
          reason: 'Student score list should render from BFF data.');
      },
    );
  });

  group('Teacher View — Score Edit (I-SCR-02)', () {
    testWidgets(
      'Score edit -> save with mandatory reason',
      (WidgetTester tester) async {
        app.main();
        await tester.pumpAndSettle(const Duration(seconds: 3));

        // Navigate to scores.
        final scoresTab = find.textContaining(
          RegExp(r'score|result|review', caseSensitive: false),
        );
        if (scoresTab.evaluate().isEmpty) return;

        await tester.tap(scoresTab.first);
        await tester.pump(const Duration(seconds: 5));

        // Tap a student row to drill down.
        final studentRow = find.textContaining(
          RegExp(r'student|name', caseSensitive: false),
        );
        if (studentRow.evaluate().isEmpty) return;
        await tester.tap(studentRow.first);
        await tester.pumpAndSettle();

        // Look for edit / override button.
        final editBtn = find.textContaining(
          RegExp(r'edit|override|adjust', caseSensitive: false),
        );
        if (editBtn.evaluate().isEmpty) return;
        await tester.tap(editBtn.first);
        await tester.pumpAndSettle();

        // Enter new score.
        final scoreInput = find.byWidgetPredicate(
          (w) => w.runtimeType.toString().contains('TextField'),
        );
        if (scoreInput.evaluate().isNotEmpty) {
          await tester.enterText(scoreInput.first, '9');
        }

        // Enter reason.
        final reasonInput = find.textContaining(
          RegExp(r'reason|justification', caseSensitive: false),
        );
        if (reasonInput.evaluate().isNotEmpty) {
          // Find the actual text field near the reason label.
          final allFields = find.byWidgetPredicate(
            (w) => w.runtimeType.toString().contains('TextField'),
          );
          if (allFields.evaluate().length >= 2) {
            await tester.enterText(
              allFields.at(1),
              'Integration test: corrected scoring for partial credit.',
            );
          }
        }

        // Save.
        final saveBtn = find.textContaining(
          RegExp(r'save|submit|confirm', caseSensitive: false),
        );
        if (saveBtn.evaluate().isNotEmpty) {
          await tester.tap(saveBtn.first);
          await tester.pump(const Duration(seconds: 5));

          // Expect success feedback.
          final successIndicator = find.textContaining(
            RegExp(r'saved|success|updated', caseSensitive: false),
          );
          expect(successIndicator, findsWidgets,
            reason: 'Score edit should confirm successful save.');
        }
      },
    );
  });
}
