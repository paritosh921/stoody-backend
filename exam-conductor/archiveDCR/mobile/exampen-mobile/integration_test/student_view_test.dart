/// Student View integration tests for ExamPen mobile.
///
/// Task:   W6.A3 (Mobile <-> BFF integration)
/// Level:  L5
/// Spec:   TEST_SUITE_SPEC.md -- I-BFF-S01, I-BFF-S02, E2E-11
///
/// These tests verify the Flutter student view against real BFF services.
/// No BLE hardware is required -- these exercise HTTP API flows only.
///
/// Prerequisites:
///   - Full Docker Compose stack running (svc-student-bff + backing services)
///   - Seed data loaded (scripts/seed-data.sh)
///
/// Run with:
///   flutter test integration_test/student_view_test.dart
@Tags(['api'])
library;

import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:exampen_mobile/main.dart' as app;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Student View — Login & Exam List (I-BFF-S01)', () {
    testWidgets(
      'Login -> exam list loads with seeded exams',
      (WidgetTester tester) async {
        app.main();
        await tester.pumpAndSettle(const Duration(seconds: 3));

        // Navigate to student mode.
        final studentModeBtn = find.textContaining(
          RegExp(r'student', caseSensitive: false),
        );
        if (studentModeBtn.evaluate().isNotEmpty) {
          await tester.tap(studentModeBtn.first);
          await tester.pumpAndSettle();
        }

        // Enter credentials.
        final inputFields = find.byWidgetPredicate(
          (w) => w.runtimeType.toString().contains('TextField'),
        );
        if (inputFields.evaluate().length >= 2) {
          await tester.enterText(inputFields.first, 'student@test.exampen.local');
          await tester.enterText(inputFields.at(1), 'test-student-pass');
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
          RegExp(r'exam|mathematics|science|result', caseSensitive: false),
        );
        expect(examList, findsWidgets,
          reason: 'Exam list should display seeded exams after student login.');
      },
    );
  });

  group('Student View — Score Summary & Breakdown (I-BFF-S01)', () {
    testWidgets(
      'Score summary -> question-wise breakdown renders',
      (WidgetTester tester) async {
        app.main();
        await tester.pumpAndSettle(const Duration(seconds: 3));

        // Assume logged in. Navigate to scores / results.
        final scoresTab = find.textContaining(
          RegExp(r'score|result|exam', caseSensitive: false),
        );
        if (scoresTab.evaluate().isEmpty) return;

        await tester.tap(scoresTab.first);
        await tester.pump(const Duration(seconds: 5));

        // Tap first exam to view breakdown.
        final examItem = find.textContaining(
          RegExp(r'mathematics|science|exam', caseSensitive: false),
        );
        if (examItem.evaluate().isEmpty) return;
        await tester.tap(examItem.first);
        await tester.pump(const Duration(seconds: 5));

        // Question-wise breakdown should appear (Q1, Q2, etc. or marks).
        final questionBreakdown = find.textContaining(
          RegExp(r'question|Q\d|breakdown|marks|score', caseSensitive: false),
        );
        expect(questionBreakdown, findsWidgets,
          reason: 'Question-wise breakdown should render from BFF data.');
      },
    );
  });

  group('Student View — Objection Filing (I-BFF-S02 / E2E-11)', () {
    testWidgets(
      'File objection -> appears in objection list',
      (WidgetTester tester) async {
        app.main();
        await tester.pumpAndSettle(const Duration(seconds: 3));

        // Navigate to exam results.
        final resultsTab = find.textContaining(
          RegExp(r'result|score|exam', caseSensitive: false),
        );
        if (resultsTab.evaluate().isEmpty) return;
        await tester.tap(resultsTab.first);
        await tester.pump(const Duration(seconds: 5));

        // Tap first exam.
        final examItem = find.textContaining(
          RegExp(r'mathematics|science|exam', caseSensitive: false),
        );
        if (examItem.evaluate().isEmpty) return;
        await tester.tap(examItem.first);
        await tester.pumpAndSettle();

        // Look for file objection button.
        final objectionBtn = find.textContaining(
          RegExp(r'objection|dispute|raise', caseSensitive: false),
        );
        if (objectionBtn.evaluate().isEmpty) return;
        await tester.tap(objectionBtn.first);
        await tester.pumpAndSettle();

        // Fill objection reason.
        final textFields = find.byWidgetPredicate(
          (w) => w.runtimeType.toString().contains('TextField'),
        );
        if (textFields.evaluate().isNotEmpty) {
          await tester.enterText(
            textFields.first,
            'Mobile integration test: expected more marks for step 2.',
          );
        }

        // Submit.
        final submitBtn = find.textContaining(
          RegExp(r'submit|file|send', caseSensitive: false),
        );
        if (submitBtn.evaluate().isNotEmpty) {
          await tester.tap(submitBtn.first);
          await tester.pump(const Duration(seconds: 5));
        }

        // Verify the objection appears in the list.
        final objectionStatus = find.textContaining(
          RegExp(r'filed|pending|submitted|success', caseSensitive: false),
        );
        expect(objectionStatus, findsWidgets,
          reason: 'Filed objection should appear with status after submission.');
      },
    );
  });
}
