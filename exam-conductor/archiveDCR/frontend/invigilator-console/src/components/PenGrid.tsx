import type { PenSyncRow, PenSyncStatus } from "@/types/api";

interface PenGridProps {
  pens: PenSyncRow[];
  /** Total pen slots to display (default 40 for a 5-dongle hub). */
  totalSlots?: number;
}

const STATUS_BG: Record<PenSyncStatus, string> = {
  pending: "bg-pen-pending",
  connecting: "bg-pen-connecting animate-pulse",
  syncing: "bg-pen-syncing animate-pulse",
  complete: "bg-pen-complete",
  failed: "bg-pen-failed",
  timeout: "bg-pen-timeout",
};

const STATUS_RING: Record<PenSyncStatus, string> = {
  pending: "ring-pen-pending/30",
  connecting: "ring-pen-connecting/40",
  syncing: "ring-pen-syncing/40",
  complete: "ring-pen-complete/40",
  failed: "ring-pen-failed/50",
  timeout: "ring-pen-timeout/40",
};

/**
 * 8-column grid of 40 pen indicators, each color-coded by sync status.
 * Empty slots render as dark placeholders.
 */
export function PenGrid({ pens, totalSlots = 40 }: PenGridProps) {
  const slots: (PenSyncRow | null)[] = [];

  // Fill with actual pens first, pad remaining with nulls
  for (let i = 0; i < totalSlots; i++) {
    slots.push(pens[i] ?? null);
  }

  return (
    <div className="grid grid-cols-8 gap-2">
      {slots.map((pen, idx) => {
        if (!pen) {
          return (
            <div
              key={`empty-${idx}`}
              className="aspect-square rounded-lg bg-gray-900 border border-gray-800 flex items-center justify-center"
            >
              <span className="text-[10px] text-gray-700">{idx + 1}</span>
            </div>
          );
        }

        const bg = STATUS_BG[pen.sync_status];
        const ring = STATUS_RING[pen.sync_status];

        return (
          <div
            key={pen.pen_mac}
            title={`${pen.pen_mac}${pen.student_id ? ` — ${pen.student_id}` : ""}\nStatus: ${pen.sync_status}`}
            className={`aspect-square rounded-lg ring-2 ${ring} flex flex-col items-center justify-center cursor-default`}
          >
            <div className={`w-4 h-4 rounded-full ${bg}`} />
            <span className="text-[9px] text-gray-400 mt-1 truncate max-w-full px-1">
              {idx + 1}
            </span>
          </div>
        );
      })}
    </div>
  );
}
