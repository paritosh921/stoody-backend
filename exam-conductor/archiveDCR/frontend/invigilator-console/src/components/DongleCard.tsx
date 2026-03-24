import type { DongleRow, DongleStatus } from "@/types/api";

interface DongleCardProps {
  dongle: DongleRow;
}

const STATUS_CONFIG: Record<
  DongleStatus,
  { bg: string; border: string; dot: string; label: string }
> = {
  healthy: {
    bg: "bg-dongle-healthy/10",
    border: "border-dongle-healthy/30",
    dot: "bg-dongle-healthy",
    label: "Healthy",
  },
  degraded: {
    bg: "bg-dongle-degraded/10",
    border: "border-dongle-degraded/30",
    dot: "bg-dongle-degraded",
    label: "Degraded",
  },
  failed: {
    bg: "bg-dongle-failed/10",
    border: "border-dongle-failed/30",
    dot: "bg-dongle-failed animate-pulse",
    label: "Failed",
  },
};

/**
 * Dongle health card showing status indicator, connected pen count, and capacity.
 */
export function DongleCard({ dongle }: DongleCardProps) {
  const config = STATUS_CONFIG[dongle.status];
  const capacity = dongle.capacity ?? 8;
  const utilization = capacity > 0
    ? Math.round((dongle.connected_pens / capacity) * 100)
    : 0;

  return (
    <div
      className={`rounded-xl border ${config.border} ${config.bg} p-4 flex flex-col gap-3`}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs text-gray-400 truncate max-w-[140px]">
          {dongle.dongle_mac}
        </span>
        <div className="flex items-center gap-1.5">
          <div className={`w-2.5 h-2.5 rounded-full ${config.dot}`} />
          <span className="text-xs font-medium text-gray-300">
            {config.label}
          </span>
        </div>
      </div>

      {/* Pen utilization */}
      <div>
        <div className="flex items-baseline justify-between mb-1">
          <span className="text-sm text-gray-300">
            {dongle.connected_pens}
            <span className="text-gray-500"> / {capacity} pens</span>
          </span>
          <span className="text-xs text-gray-500">{utilization}%</span>
        </div>
        <div className="h-1.5 rounded-full bg-gray-800 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ${config.dot.replace("animate-pulse", "")}`}
            style={{ width: `${utilization}%` }}
          />
        </div>
      </div>
    </div>
  );
}
