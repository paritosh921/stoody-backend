import type { PenSyncRow } from "@/types/api";

interface PenProgressBarProps {
  pen: PenSyncRow;
}

const STATUS_COLORS: Record<string, string> = {
  pending: "bg-pen-pending",
  connecting: "bg-pen-connecting",
  syncing: "bg-pen-syncing",
  complete: "bg-pen-complete",
  failed: "bg-pen-failed",
  timeout: "bg-pen-timeout",
};

const STATUS_LABELS: Record<string, string> = {
  pending: "Pending",
  connecting: "Connecting",
  syncing: "Syncing",
  complete: "Complete",
  failed: "Failed",
  timeout: "Timeout",
};

/**
 * Single pen sync progress bar showing MAC, student, status, and chunk progress.
 */
export function PenProgressBar({ pen }: PenProgressBarProps) {
  const barColor = STATUS_COLORS[pen.sync_status] ?? "bg-gray-500";
  const label = STATUS_LABELS[pen.sync_status] ?? pen.sync_status;

  const percent =
    pen.total_chunks && pen.total_chunks > 0 && pen.bytes_received != null
      ? Math.min(100, Math.round((pen.bytes_received / pen.total_chunks) * 100))
      : pen.sync_status === "complete"
        ? 100
        : 0;

  return (
    <div className="rounded-lg bg-gray-900 p-3">
      <div className="flex items-center justify-between text-xs mb-1.5">
        <span className="font-mono text-gray-400 truncate max-w-[120px]">
          {pen.pen_mac}
        </span>
        {pen.student_id && (
          <span className="text-gray-500 truncate max-w-[100px]">
            {pen.student_id}
          </span>
        )}
        <span
          className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium text-gray-950 ${barColor}`}
        >
          {label}
        </span>
      </div>

      {/* Progress track */}
      <div className="h-2 rounded-full bg-gray-800 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${barColor}`}
          style={{ width: `${percent}%` }}
        />
      </div>

      {pen.total_chunks != null && (
        <p className="text-[10px] text-gray-500 mt-1 text-right">
          {pen.bytes_received ?? 0} / {pen.total_chunks} chunks
        </p>
      )}
    </div>
  );
}
