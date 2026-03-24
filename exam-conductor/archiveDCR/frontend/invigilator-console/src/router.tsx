import { createBrowserRouter, Navigate } from "react-router-dom";
import { InvigLayout } from "./layouts/InvigLayout";
import { SessionListPage } from "./pages/SessionListPage";
import { DashboardPage } from "./pages/DashboardPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <InvigLayout />,
    children: [
      { index: true, element: <Navigate to="/sessions" replace /> },
      { path: "sessions", element: <SessionListPage /> },
      { path: "sessions/:sessionId", element: <DashboardPage /> },
    ],
  },
]);
