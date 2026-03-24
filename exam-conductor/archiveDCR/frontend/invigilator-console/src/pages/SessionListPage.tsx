import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

interface SessionListItem {
  exam_id: string;
  state: string;
  timer_remaining_sec: number;
  upload_status: string;
}

const STATE_BADGE: Record<string, { bg: string; text: string }> = {
  running: { bg: "bg-green-900/60", text: "text-green-300" },
  paused: { bg: "bg-yellow-900/60", text: "text-yellow-300" },
  completed: { bg: "bg-blue-900/60", text: "text-blue-300" },
  cancelled: { bg: "bg-gray-800", text: "text-gray-400" },
};

const UPLOAD_BADGE: Record<string, { bg: string; text: string }> = {
  pending: { bg: "bg-gray-800", text: "text-gray-400" },
  in_progress: { bg: "bg-blue-900/60", text: "text-blue-300" },
  complete: { bg: "bg-green-900/60", text: "text-green-300" },
  partial: { bg: "bg-yellow-900/60", text: "text-yellow-300" },
};

/**
 * Displays active exam sessions fetched from the REST API.
 * Each row links to the real-time dashboard for that session.
 */
export function SessionListPage() {
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchSessions() {
      try {
        const res = await fetch("/api/v1/invigilator/sessions");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as SessionListItem[];
        if (!cancelled) {
          setSessions(data);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load sessions");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchSessions();
    return () => { cancelled = true; };
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">Loading sessions...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-red-400">Error: {error}</p>
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">No active exam sessions</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-xl font-semibold mb-6">Exam Sessions</h1>

      <div className="grid gap-3">
        {sessions.map((s) => {
          const stateBadge = STATE_BADGE[s.state] ?? STATE_BADGE.cancelled;
          const uploadBadge = UPLOAD_BADGE[s.upload_status] ?? UPLOAD_BADGE.pending;

          return (
            <Link
              key={s.exam_id}
              to={`/sessions/${s.exam_id}`}
              className="flex items-center justify-between rounded-xl bg-gray-900 border border-gray-800 hover:border-gray-700 transition-colors px-5 py-4"
            >
              <div className="flex items-center gap-4">
                <span className="font-mono text-sm text-gray-300">
                  {s.exam_id}
                </span>
                <span
                  className={`inline-flex rounded-full px-2.5 py-0.5 text-xs font-medium ${stateBadge?.bg ?? ''} ${stateBadge?.text ?? ''}`}
                >
                  {s.state}
                </span>
              </div>

              <div className="flex items-center gap-4">
                <span
                  className={`inline-flex rounded-full px-2.5 py-0.5 text-xs font-medium ${uploadBadge?.bg ?? ''} ${uploadBadge?.text ?? ''}`}
                >
                  {s.upload_status}
                </span>
                <span className="text-sm text-gray-500">
                  {formatTime(s.timer_remaining_sec)}
                </span>
                <span className="text-gray-600">&rarr;</span>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}

function formatTime(totalSec: number): string {
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${pad(h)}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}
