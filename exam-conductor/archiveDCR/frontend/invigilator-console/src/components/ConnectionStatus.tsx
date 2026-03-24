import type { ConnectionState } from "@/types/ws";
import type { SessionSummary } from "@/types/api";

interface ConnectionStatusProps {
  connectionState: ConnectionState;
  session: SessionSummary | null;
}

interface Indicator {
  label: string;
  status: "green" | "yellow" | "red" | "gray";
}

const DOT_COLOR: Record<Indicator["status"], string> = {
  green: "bg-green-400",
  yellow: "bg-yellow-400 animate-pulse",
  red: "bg-red-400 animate-pulse",
  gray: "bg-gray-600",
};

/**
 * Hub / WiFi / Backend connectivity indicators.
 * Derives WiFi and backend status from session data and WS connection state.
 */
export function ConnectionStatus({
  connectionState,
  session,
}: ConnectionStatusProps) {
  const indicators: Indicator[] = [
    // WebSocket connection to svc-invig-console
    {
      label: "Hub WS",
      status: wsToIndicator(connectionState),
    },
    // Backend reachability (inferred from session snapshot presence + recency)
    {
      label: "Backend",
      status: backendIndicator(session),
    },
    // Upload channel status
    {
      label: "Upload",
      status: uploadIndicator(session),
    },
  ];

  return (
    <div className="flex items-center gap-4">
      {indicators.map((ind) => (
        <div key={ind.label} className="flex items-center gap-1.5">
          <div className={`w-2 h-2 rounded-full ${DOT_COLOR[ind.status]}`} />
          <span className="text-xs text-gray-400">{ind.label}</span>
        </div>
      ))}
    </div>
  );
}

function wsToIndicator(state: ConnectionState): Indicator["status"] {
  switch (state) {
    case "connected":
      return "green";
    case "connecting":
      return "yellow";
    case "disconnected":
    case "error":
      return "red";
  }
}

function backendIndicator(session: SessionSummary | null): Indicator["status"] {
  if (!session) return "gray";
  if (!session.backend_seen_at) return "yellow";

  const seenAt = new Date(session.backend_seen_at).getTime();
  const ageSec = (Date.now() - seenAt) / 1000;
  if (ageSec < 30) return "green";
  if (ageSec < 120) return "yellow";
  return "red";
}

function uploadIndicator(session: SessionSummary | null): Indicator["status"] {
  if (!session) return "gray";
  switch (session.upload_status) {
    case "complete":
      return "green";
    case "in_progress":
      return "yellow";
    case "partial":
      return "yellow";
    case "pending":
      return "gray";
  }
}
