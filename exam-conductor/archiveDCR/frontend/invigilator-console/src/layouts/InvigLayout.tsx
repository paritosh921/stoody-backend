import { Outlet, Link, useLocation } from "react-router-dom";

export function InvigLayout() {
  const location = useLocation();
  const isSessionList = location.pathname === "/sessions";

  return (
    <div className="min-h-screen flex flex-col bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 bg-gray-900 border-b border-gray-800">
        <Link to="/sessions" className="flex items-center gap-2">
          <span className="text-lg font-semibold tracking-tight text-white">
            ExamPen Invigilator
          </span>
        </Link>

        {!isSessionList && (
          <Link
            to="/sessions"
            className="text-sm text-gray-400 hover:text-gray-200 transition-colors"
          >
            &larr; Back to Sessions
          </Link>
        )}
      </header>

      {/* Content area */}
      <main className="flex-1 p-6">
        <Outlet />
      </main>
    </div>
  );
}
