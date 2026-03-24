// ---------------------------------------------------------------------------
// ConfidenceBar — horizontal bar showing AI confidence level (0-1).
// ---------------------------------------------------------------------------

interface ConfidenceBarProps {
  value: number; // 0..1
  showLabel?: boolean;
  className?: string;
}

function getColor(value: number): string {
  if (value >= 0.8) return 'bg-green-500';
  if (value >= 0.5) return 'bg-yellow-500';
  return 'bg-red-500';
}

export function ConfidenceBar({
  value,
  showLabel = true,
  className = '',
}: ConfidenceBarProps) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <div className="h-2 flex-1 rounded-full bg-gray-200">
        <div
          className={`h-2 rounded-full transition-all ${getColor(value)}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {showLabel && (
        <span className="text-xs tabular-nums text-gray-600">{pct}%</span>
      )}
    </div>
  );
}
