import { createBrowserRouter, Navigate } from "react-router-dom";
import StudentLayout from "@/layouts/StudentLayout";
import UpcomingExamsPage from "@/pages/UpcomingExamsPage";
import PastExamsPage from "@/pages/PastExamsPage";
import ScoreSummaryPage from "@/pages/ScoreSummaryPage";
import QuestionBreakdownPage from "@/pages/QuestionBreakdownPage";
import AnswerViewerPage from "@/pages/AnswerViewerPage";
import ObjectionFilePage from "@/pages/ObjectionFilePage";
import ObjectionStatusPage from "@/pages/ObjectionStatusPage";
import ChatPage from "@/pages/ChatPage";
import PerformancePage from "@/pages/PerformancePage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <StudentLayout />,
    children: [
      { index: true, element: <Navigate to="/exams/upcoming" replace /> },
      { path: "exams/upcoming", element: <UpcomingExamsPage /> },
      { path: "exams/past", element: <PastExamsPage /> },
      { path: "scores/:examId", element: <ScoreSummaryPage /> },
      { path: "scores/:examId/questions", element: <QuestionBreakdownPage /> },
      {
        path: "scores/:examId/answers/:questionId",
        element: <AnswerViewerPage />,
      },
      { path: "objections/file", element: <ObjectionFilePage /> },
      { path: "objections/status", element: <ObjectionStatusPage /> },
      { path: "chat/:examId/:teacherId", element: <ChatPage /> },
      { path: "performance", element: <PerformancePage /> },
    ],
  },
]);
