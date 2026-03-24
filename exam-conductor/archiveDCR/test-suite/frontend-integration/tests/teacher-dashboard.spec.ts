/**
 * Teacher Dashboard <-> svc-teacher-bff integration tests.
 *
 * Task:   W6.A2 (frontend integration)
 * Level:  L5
 * Spec:   TEST_SUITE_SPEC.md -- I-BFF-T01, I-BFF-T02, E2E-10
 *
 * Prerequisites:
 *   - Full Docker Compose stack running (all BFF + backing services)
 *   - Seed data loaded (scripts/seed-data.sh)
 *   - teacher-dashboard dev server on TEACHER_BASE_URL (default 5173)
 */

import { test, expect, Page } from "@playwright/test";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TEACHER_BFF_URL =
  process.env.TEACHER_BFF_URL ?? "http://localhost:8010";

/** Seed credentials -- matches scripts/seed-data.sh defaults. */
const TEACHER_CREDS = {
  email: "teacher@test.exampen.local",
  password: "test-teacher-pass",
};

const STUDENT_CREDS = {
  email: "student@test.exampen.local",
  password: "test-student-pass",
};

/**
 * Perform login via the UI login form.
 * Assumes a standard email + password form on /login.
 */
async function loginAs(
  page: Page,
  creds: { email: string; password: string },
): Promise<void> {
  await page.goto("/login");
  await page.getByLabel(/email/i).fill(creds.email);
  await page.getByLabel(/password/i).fill(creds.password);
  await page.getByRole("button", { name: /sign in|log in|login/i }).click();
  // Wait for redirect away from /login.
  await page.waitForURL((url) => !url.pathname.includes("/login"), {
    timeout: 15_000,
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe("Teacher Dashboard — Login & Exam List", () => {
  test(
    "I-BFF-T01: Login -> exam list loads with seeded exams",
    async ({ page }) => {
      await loginAs(page, TEACHER_CREDS);

      // After login, expect to land on exam list / dashboard.
      await expect(
        page.getByRole("heading", { name: /exam|dashboard/i }),
      ).toBeVisible();

      // At least one exam should be present from seeded data.
      const examRows = page.locator('[data-testid="exam-row"], tr, .exam-card');
      await expect(examRows.first()).toBeVisible({ timeout: 10_000 });
    },
  );

  test(
    "Create exam -> appears in exam list",
    async ({ page }) => {
      await loginAs(page, TEACHER_CREDS);

      // Navigate to create exam page.
      await page.getByRole("link", { name: /create|new exam/i }).click();
      await expect(page).toHaveURL(/create|new/i);

      // Fill minimal exam creation form.
      const examName = `Integration Test Exam ${Date.now()}`;
      await page.getByLabel(/exam name|title/i).fill(examName);
      await page.getByLabel(/subject/i).selectOption({ index: 1 });
      await page.getByLabel(/class|grade/i).selectOption({ index: 1 });
      await page
        .getByRole("button", { name: /create|save|submit/i })
        .click();

      // Expect redirect to exam list or detail page.
      await page.waitForURL((url) => !url.pathname.includes("create"), {
        timeout: 15_000,
      });

      // Navigate back to exam list if needed.
      if (!page.url().includes("exam")) {
        await page.goto("/exams");
      }

      // The newly created exam must appear.
      await expect(page.getByText(examName)).toBeVisible({ timeout: 10_000 });
    },
  );
});

test.describe("Teacher Dashboard — Score Review", () => {
  test(
    "E2E-10: Class scores page renders data from BFF",
    async ({ page }) => {
      await loginAs(page, TEACHER_CREDS);

      // Navigate to scores / class overview.
      await page.getByRole("link", { name: /score|review|class/i }).click();

      // Verify score table or grid is present.
      const scoreContainer = page.locator(
        '[data-testid="score-table"], table, .score-grid',
      );
      await expect(scoreContainer).toBeVisible({ timeout: 10_000 });

      // At least one student row should be visible from seeded data.
      const studentRows = page.locator(
        '[data-testid="student-score-row"], tbody tr, .student-row',
      );
      await expect(studentRows.first()).toBeVisible();
    },
  );

  test(
    "Score override requires mandatory reason (I-SCR-02)",
    async ({ page }) => {
      await loginAs(page, TEACHER_CREDS);

      // Navigate to a score detail page.
      await page.getByRole("link", { name: /score|review/i }).click();

      // Click first student to drill down.
      const firstStudent = page.locator(
        '[data-testid="student-score-row"], tbody tr, .student-row',
      );
      await firstStudent.first().click();

      // Find and click an edit/override button.
      const overrideBtn = page.getByRole("button", {
        name: /edit|override|adjust/i,
      });
      if (await overrideBtn.isVisible()) {
        await overrideBtn.click();

        // Change the score value.
        const scoreInput = page.getByLabel(/score|marks|points/i);
        await scoreInput.clear();
        await scoreInput.fill("8");

        // Try to submit without a reason.
        const submitBtn = page.getByRole("button", {
          name: /save|submit|confirm/i,
        });
        await submitBtn.click();

        // Expect a validation error about the reason.
        await expect(
          page.getByText(/reason|justification|required/i),
        ).toBeVisible();
      }
    },
  );
});

test.describe("Teacher Dashboard — RBAC Enforcement", () => {
  test(
    "I-BFF-T02: Student token -> redirect or 403 on teacher routes",
    async ({ page }) => {
      // Login as a student.
      await loginAs(page, STUDENT_CREDS);

      // Attempt to navigate to a teacher-only route.
      const response = await page.goto("/exams");

      // Should either redirect to /login, show 403, or show an access denied
      // message. The exact behavior depends on the auth guard implementation.
      const url = page.url();
      const is403 = response?.status() === 403;
      const redirectedToLogin = url.includes("/login");
      const accessDenied = await page
        .getByText(/access denied|unauthorized|forbidden/i)
        .isVisible()
        .catch(() => false);

      expect(is403 || redirectedToLogin || accessDenied).toBeTruthy();
    },
  );
});

test.describe("Teacher Dashboard — Objection Inbox", () => {
  test(
    "Objection inbox loads and resolve flow works (I-REV-01)",
    async ({ page }) => {
      await loginAs(page, TEACHER_CREDS);

      // Navigate to objection inbox.
      await page.getByRole("link", { name: /objection|review/i }).click();

      // Objection list should render.
      const objectionList = page.locator(
        '[data-testid="objection-list"], .objection-card, tbody tr',
      );
      await expect(objectionList.first()).toBeVisible({ timeout: 10_000 });

      // Click first objection to view detail.
      await objectionList.first().click();

      // Expect detail view with student answer and objection text.
      await expect(
        page.getByText(/objection|reason|student/i),
      ).toBeVisible();

      // Resolve (approve or reject) the objection.
      const resolveBtn = page.getByRole("button", {
        name: /approve|reject|resolve/i,
      });
      if (await resolveBtn.isVisible()) {
        // If reject, fill reason.
        await resolveBtn.click();

        const reasonField = page.getByLabel(/reason|comment/i);
        if (await reasonField.isVisible()) {
          await reasonField.fill("Integration test resolution reason.");
          await page
            .getByRole("button", { name: /confirm|submit/i })
            .click();
        }

        // Expect status update.
        await expect(
          page.getByText(/resolved|approved|rejected/i),
        ).toBeVisible({ timeout: 10_000 });
      }
    },
  );
});
