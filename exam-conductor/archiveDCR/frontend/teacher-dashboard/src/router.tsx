// ---------------------------------------------------------------------------
// AppRouter — React Router v6 route definitions for the teacher dashboard.
// ---------------------------------------------------------------------------

import { Routes, Route, Navigate } from 'react-router-dom';
import { TeacherLayout } from '@/layouts/TeacherLayout';
import { ExamListPage } from '@/pages/ExamListPage';
import { ExamDetailPage } from '@/pages/ExamDetailPage';
import { CreateExamPage } from '@/pages/CreateExamPage';
import { RubricEditorPage } from '@/pages/RubricEditorPage';
import { QuestionRegionEditorPage } from '@/pages/QuestionRegionEditorPage';
import { ClassScorePage } from '@/pages/ClassScorePage';
import { StudentDrilldownPage } from '@/pages/StudentDrilldownPage';
import { ObjectionInboxPage } from '@/pages/ObjectionInboxPage';
import { ObjectionDetailPage } from '@/pages/ObjectionDetailPage';
import { PlagiarismReviewPage } from '@/pages/PlagiarismReviewPage';
import { LeaderboardPage } from '@/pages/LeaderboardPage';
import { ClassAnalyticsPage } from '@/pages/ClassAnalyticsPage';

export function AppRouter() {
  return (
    <Routes>
      <Route element={<TeacherLayout />}>
        {/* Exams */}
        <Route path="/exams" element={<ExamListPage />} />
        <Route path="/exams/create" element={<CreateExamPage />} />
        <Route path="/exams/:examId" element={<ExamDetailPage />} />
        <Route path="/exams/:examId/rubric" element={<RubricEditorPage />} />

        {/* Question Regions */}
        <Route
          path="/question-regions/:examId"
          element={<QuestionRegionEditorPage />}
        />

        {/* Scores */}
        <Route path="/scores/:examId" element={<ClassScorePage />} />
        <Route
          path="/scores/:examId/:studentId"
          element={<StudentDrilldownPage />}
        />

        {/* Objections */}
        <Route
          path="/objections/:examId"
          element={<ObjectionInboxPage />}
        />
        <Route
          path="/objections/detail/:objectionId"
          element={<ObjectionDetailPage />}
        />

        {/* Analytics */}
        <Route path="/analytics" element={<ClassAnalyticsPage />} />
        <Route
          path="/analytics/:examId/leaderboard"
          element={<LeaderboardPage />}
        />

        {/* Plagiarism */}
        <Route path="/plagiarism/:examId" element={<PlagiarismReviewPage />} />

        {/* Default redirect */}
        <Route path="/" element={<Navigate to="/exams" replace />} />
        <Route path="*" element={<Navigate to="/exams" replace />} />
      </Route>
    </Routes>
  );
}
