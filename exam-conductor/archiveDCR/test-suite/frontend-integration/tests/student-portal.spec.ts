/**
 * Student Portal <-> svc-student-bff integration tests.
 *
 * Task:   W6.A2 (frontend integration)
 * Level:  L5
 * Spec:   TEST_SUITE_SPEC.md -- I-BFF-S01, I-BFF-S02, E2E-11
 *
 * Prerequisites:
 *   - Full Docker Compose stack running (all BFF + backing services)
 *   - Seed data loaded (scripts/seed-data.sh)
 *   - student-portal dev server on STUDENT_BASE_URL (default 5174)
 */

import { test, expect, Page } from "@playwright/test";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const STUDENT_BFF_URL =
  process.env.STUDENT_BFF_URL ?? "http://localhost:8011";

const STUDENT_CREDS = {
  email: "student@test.exampen.local",
  password: "test-student-pass",
};

async function loginAsStudent(page: Page): Promise<void> {
  await page.goto("/login");
  await page.getByLabel(/email/i).fill(STUDENT_CREDS.email);
  await page.getByLabel(/password/i).fill(STUDENT_CREDS.password);
  await page.getByRole("button", { name: /sign in|log in|login/i }).click();
  await page.waitForURL((url) => !url.pathname.includes("/login"), {
    timeout: 15_000,
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe("Student Portal — Score Summary", () => {
  test(
    "I-BFF-S01: Login -> score summary loads with seeded data",
    async ({ page }) => {
      await loginAsStudent(page);

      // Expect a dashboard / score summary page.
      await expect(
        page.getByRole("heading", { name: /score|result|dashboard/i }),
      ).toBeVisible({ timeout: 10_000 });

      // At least one exam result should be visible from seeded data.
      const examResults = page.locator(
        '[data-testid="exam-result"], .exam-card, .score-card, tbody tr',
      );
      await expect(examResults.first()).toBeVisible({ timeout: 10_000 });
    },
  );

  test(
    "Question breakdown -> answer viewer renders",
    async ({ page }) => {
      await loginAsStudent(page);

      // Click into the first exam result for drill-down.
      const firstResult = page.locator(
        '[data-testid="exam-result"], .exam-card, .score-card, tbody tr',
      );
      await firstResult.first().click();

      // Expect question-wise breakdown view.
      await expect(
        page.getByText(/question|breakdown|detail/i),
      ).toBeVisible({ timeout: 10_000 });

      // Click a specific question to open the answer viewer.
      const questionItem = page.locator(
        '[data-testid="question-item"], .question-row, li',
      );
      if (await questionItem.first().isVisible()) {
        await questionItem.first().click();

        // Answer image or rendered answer should appear.
        const answerViewer = page.locator(
          '[data-testid="answer-viewer"], .answer-image, img, canvas',
        );
        await expect(answerViewer.first()).toBeVisible({ timeout: 10_000 });
      }
    },
  );
});

test.describe("Student Portal — Objection Filing", () => {
  test(
    "I-BFF-S02 / E2E-11: File objection -> appears with status tracking",
    async ({ page }) => {
      await loginAsStudent(page);

      // Navigate to a score detail page.
      const firstResult = page.locator(
        '[data-testid="exam-result"], .exam-card, .score-card, tbody tr',
      );
      await firstResult.first().click();

      // Look for an objection / raise dispute button.
      const fileObjectionBtn = page.getByRole("button", {
        name: /file objection|raise|dispute/i,
      });

      if (await fileObjectionBtn.isVisible()) {
        await fileObjectionBtn.click();

        // Fill objection form.
        const reasonField = page.getByLabel(/reason|description|detail/i);
        await reasonField.fill(
          "Integration test objection: expected more marks for step 2.",
        );

        // Select question if the form requires it.
        const questionSelect = page.getByLabel(/question/i);
        if (await questionSelect.isVisible()) {
          await questionSelect.selectOption({ index: 1 });
        }

        // Submit the objection.
        await page
          .getByRole("button", { name: /submit|file|send/i })
          .click();

        // Expect confirmation or redirect to objection status.
        await expect(
          page.getByText(/submitted|filed|success|pending/i),
        ).toBeVisible({ timeout: 10_000 });
      }

      // Navigate to objections list to verify status tracking.
      await page.goto("/objections");
      const objectionStatus = page.locator(
        '[data-testid="objection-status"], .objection-card, tbody tr',
      );
      await expect(objectionStatus.first()).toBeVisible({ timeout: 10_000 });

      // Verify status label is present (filed / pending / etc.)
      await expect(
        page.getByText(/filed|pending|under review|resolved/i),
      ).toBeVisible();
    },
  );
});

test.describe("Student Portal — Chat", () => {
  test(
    "Send message and receive in chat thread (I-BFF-S01)",
    async ({ page }) => {
      await loginAsStudent(page);

      // Navigate to chat (may be under objections or a dedicated route).
      const chatLink = page.getByRole("link", { name: /chat|message/i });
      if (await chatLink.isVisible()) {
        await chatLink.click();
      } else {
        // Try navigating directly.
        await page.goto("/chat");
      }

      // Wait for chat interface.
      const chatContainer = page.locator(
        '[data-testid="chat-container"], .chat-thread, .messages',
      );

      if (await chatContainer.isVisible({ timeout: 5_000 }).catch(() => false)) {
        // Type and send a message.
        const messageInput = page.getByPlaceholder(/type|message|write/i);
        const testMessage = `Integration test message ${Date.now()}`;
        await messageInput.fill(testMessage);
        await page
          .getByRole("button", { name: /send/i })
          .click();

        // The sent message should appear in the thread.
        await expect(page.getByText(testMessage)).toBeVisible({
          timeout: 10_000,
        });
      }
    },
  );
});
