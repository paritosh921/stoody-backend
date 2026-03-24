// ---------------------------------------------------------------------------
// TeacherLayout — sidebar navigation + header with user info.
// ---------------------------------------------------------------------------

import { NavLink, Outlet } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';

interface NavItem {
  to: string;
  label: string;
  icon: string;
}

const NAV_ITEMS: NavItem[] = [
  { to: '/exams', label: 'Exams', icon: 'clipboard-list' },
  { to: '/scores', label: 'Scores', icon: 'table-cells' },
  { to: '/objections', label: 'Objections', icon: 'chat-bubble-left' },
  { to: '/analytics', label: 'Analytics', icon: 'chart-bar' },
  { to: '/plagiarism', label: 'Plagiarism', icon: 'shield-exclamation' },
];

function NavIcon({ name }: { name: string }) {
  // Minimal SVG icons keyed by name; avoids an icon library dependency.
  const paths: Record<string, string> = {
    'clipboard-list':
      'M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 000-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8h6M7 12h4',
    'table-cells':
      'M3 3h18v18H3V3zm0 6h18M3 15h18M9 3v18M15 3v18',
    'chat-bubble-left':
      'M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z',
    'chart-bar':
      'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6m6 0h6m-6 0V9a2 2 0 012-2h2a2 2 0 012 2v10m6 0v-4a2 2 0 00-2-2h-2a2 2 0 00-2 2v4',
    'shield-exclamation':
      'M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z',
  };

  return (
    <svg
      className="h-5 w-5 shrink-0"
      fill="none"
      viewBox="0 0 24 24"
      strokeWidth={1.5}
      stroke="currentColor"
    >
      <path strokeLinecap="round" strokeLinejoin="round" d={paths[name] ?? ''} />
    </svg>
  );
}

export function TeacherLayout() {
  const { displayName, isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-2">ExamPen Teacher Dashboard</h1>
          <p className="text-gray-600 mb-4">
            Please authenticate via Stoody to access the dashboard.
          </p>
          <a
            href="/"
            className="text-brand-600 hover:text-brand-700 underline text-sm"
          >
            Return to Stoody
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="flex w-56 shrink-0 flex-col border-r border-gray-200 bg-white">
        <div className="flex h-14 items-center border-b border-gray-200 px-4">
          <span className="text-lg font-bold text-brand-600">ExamPen</span>
        </div>

        <nav className="flex-1 space-y-1 px-2 py-3">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition ${
                  isActive
                    ? 'bg-brand-50 text-brand-700'
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }`
              }
            >
              <NavIcon name={item.icon} />
              {item.label}
            </NavLink>
          ))}
        </nav>

        {/* User info */}
        <div className="border-t border-gray-200 px-4 py-3">
          <p className="truncate text-sm font-medium text-gray-900">
            {displayName}
          </p>
          <p className="text-xs text-gray-500">Teacher</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-gray-50 p-6">
        <Outlet />
      </main>
    </div>
  );
}
