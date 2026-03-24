import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "@/hooks/useAuth";
import clsx from "clsx";

interface NavItem {
  label: string;
  to: string;
  icon: string;
}

const NAV_ITEMS: NavItem[] = [
  { label: "Upcoming", to: "/exams/upcoming", icon: "📋" },
  { label: "Past Exams", to: "/exams/past", icon: "📄" },
  { label: "Objections", to: "/objections/status", icon: "✋" },
  { label: "Performance", to: "/performance", icon: "📊" },
];

function SidebarLink({ item }: { item: NavItem }) {
  return (
    <NavLink
      to={item.to}
      className={({ isActive }) =>
        clsx(
          "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
          isActive
            ? "bg-primary-100 text-primary-700"
            : "text-gray-600 hover:bg-gray-100 hover:text-gray-900",
        )
      }
    >
      <span className="text-base">{item.icon}</span>
      {item.label}
    </NavLink>
  );
}

export default function StudentLayout() {
  const { user, isAuthenticated } = useAuth();

  if (!isAuthenticated || !user) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-2">
            ExamPen Student Portal
          </h1>
          <p className="text-gray-600 mb-4">
            Please authenticate via Stoody to view your exam results.
          </p>
          <a
            href="/"
            className="text-primary-600 hover:text-primary-700 underline text-sm"
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
      <aside className="flex w-56 flex-col border-r border-gray-200 bg-white">
        <div className="flex h-14 items-center border-b border-gray-200 px-4">
          <span className="text-lg font-semibold text-primary-700">
            ExamPen
          </span>
        </div>

        <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4">
          {NAV_ITEMS.map((item) => (
            <SidebarLink key={item.to} item={item} />
          ))}
        </nav>

        <div className="border-t border-gray-200 px-4 py-3">
          <p className="truncate text-sm font-medium text-gray-700">
            {user.name}
          </p>
          <p className="text-xs text-gray-400">{user.role}</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-gray-50 p-6">
        <Outlet />
      </main>
    </div>
  );
}
