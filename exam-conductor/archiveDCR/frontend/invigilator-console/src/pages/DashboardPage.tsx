import { useParams, useSearchParams } from "react-router-dom";
import { useExamState } from "@/hooks/useExamState";
import { TimerDisplay } from "@/components/TimerDisplay";
import { PenGrid } from "@/components/PenGrid";
import { PenProgressBar } from "@/components/PenProgressBar";
import { DongleCard } from "@/components/DongleCard";
import { AlertBanner } from "@/components/AlertBanner";
import { ConnectionStatus } from "@/components/ConnectionStatus";

const AUTH_TOKEN_STORAGE_KEY = "exampen_auth_token";

/**
 * Read the JWT auth token from (in priority order):
 *   1. `?token=` query parameter (set when Stoody embeds this page)
 *   2. localStorage (persisted from a previous visit)
 *
 * When a query-param token is found it is persisted to localStorage
 * so subsequent navigations within the SPA don't lose it.
 */
function useAuthToken(): string | null {
  const [searchParams] = useSearchParams();
  const qpToken = searchParams.get("token");

  if (qpToken) {
    try {
      localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, qpToken);
    } catch {
      // localStorage may be unavailable in some contexts; ignore.
    }
    return qpToken;
  }

  try {
    return localStorage.getItem(AUTH_TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

/**
 * Real-time exam dashboard. All data flows from the useExamState hook
 * which aggregates WebSocket events from svc-invig-console.
 *
 * Auth token is read from `?token=` query param or localStorage.
 * The sessionId URL param identifies the exam to subscribe to.
 */
export function DashboardPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const authToken = useAuthToken();
  const examId = sessionId ?? null;

  const {
    session,
    pens,
    dongles,
    uploadProgress,
    alerts,
    connectionState,
    dismissAlert,
  } = useExamState(authToken, examId);

  const activePens = pens.filter(
    (p) => p.sync_status === "syncing" || p.sync_status === "connecting",
  );

  return (
    <div className="max-w-7xl mx-auto flex flex-col gap-6">
      {/* Top bar: timer + connection + session state */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-4">
          <TimerDisplay
            remainingSeconds={session?.timer_remaining_sec ?? 0}
          />
          {session && (
            <span className="text-sm font-medium text-gray-400 uppercase tracking-wide">
              {session.state}
            </span>
          )}
        </div>

        <ConnectionStatus
          connectionState={connectionState}
          session={session}
        />
      </div>

      {/* Alerts */}
      <AlertBanner alerts={alerts} onDismiss={dismissAlert} />

      {/* Main grid: left = pen grid + upload, right = dongles + active syncs */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column: Pen grid (spans 2 cols on lg) */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          {/* Pen grid */}
          <Section title="Pen Overview">
            <PenGrid pens={pens} />
            <PenLegend />
          </Section>

          {/* Upload progress */}
          {uploadProgress && (
            <Section title="Upload Progress">
              <UploadBar
                uploaded={uploadProgress.uploaded_chunks}
                total={uploadProgress.total_chunks}
                status={uploadProgress.status}
              />
            </Section>
          )}
        </div>

        {/* Right column: dongles + active syncs */}
        <div className="flex flex-col gap-6">
          {/* Dongles */}
          <Section title={`Dongles (${dongles.length})`}>
            {dongles.length === 0 ? (
              <p className="text-sm text-gray-600">
                No dongle data received yet
              </p>
            ) : (
              <div className="flex flex-col gap-3">
                {dongles.map((d) => (
                  <DongleCard key={d.dongle_mac} dongle={d} />
                ))}
              </div>
            )}
          </Section>

          {/* Active pen syncs */}
          <Section title={`Active Syncs (${activePens.length})`}>
            {activePens.length === 0 ? (
              <p className="text-sm text-gray-600">No active syncs</p>
            ) : (
              <div className="flex flex-col gap-2">
                {activePens.map((p) => (
                  <PenProgressBar key={p.pen_mac} pen={p} />
                ))}
              </div>
            )}
          </Section>
        </div>
      </div>
    </div>
  );
}

/* ---- Helper sub-components (kept local to stay under 250 lines) ---- */

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-5">
      <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wide mb-4">
        {title}
      </h2>
      {children}
    </div>
  );
}

function PenLegend() {
  const items = [
    { color: "bg-pen-pending", label: "Pending" },
    { color: "bg-pen-connecting", label: "Connecting" },
    { color: "bg-pen-syncing", label: "Syncing" },
    { color: "bg-pen-complete", label: "Complete" },
    { color: "bg-pen-failed", label: "Failed" },
    { color: "bg-pen-timeout", label: "Timeout" },
  ];

  return (
    <div className="flex flex-wrap gap-3 mt-4">
      {items.map((item) => (
        <div key={item.label} className="flex items-center gap-1.5">
          <div className={`w-3 h-3 rounded-full ${item.color}`} />
          <span className="text-[11px] text-gray-500">{item.label}</span>
        </div>
      ))}
    </div>
  );
}

function UploadBar({
  uploaded,
  total,
  status,
}: {
  uploaded: number;
  total: number;
  status: string;
}) {
  const percent = total > 0 ? Math.round((uploaded / total) * 100) : 0;
  const barColor =
    status === "complete" ? "bg-green-500" : "bg-blue-500";

  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <span className="text-sm text-gray-300">
          {uploaded} / {total} chunks
        </span>
        <span className="text-xs text-gray-500">
          {percent}% - {status}
        </span>
      </div>
      <div className="h-2.5 rounded-full bg-gray-800 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${barColor}`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
