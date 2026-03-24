import clsx from "clsx";

interface PercentileChartProps {
  percentile: number;
}

/**
 * Visual bar showing the student's percentile position (0-100).
 * The marker slides along a gradient bar.
 */
export default function PercentileChart({ percentile }: PercentileChartProps) {
  const clamped = Math.max(0, Math.min(100, percentile));

  const color =
    clamped >= 75
      ? "text-green-600"
      : clamped >= 50
        ? "text-yellow-600"
        : "text-red-600";

  return (
    <div className="w-full">
      {/* Labels */}
      <div className="mb-1 flex justify-between text-xs text-gray-400">
        <span>0th</span>
        <span>50th</span>
        <span>100th</span>
      </div>

      {/* Bar */}
      <div className="relative h-4 w-full overflow-hidden rounded-full bg-gradient-to-r from-red-300 via-yellow-300 to-green-400">
        {/* Marker */}
        <div
          className="absolute top-0 h-full w-1 -translate-x-1/2 rounded bg-gray-900"
          style={{ left: `${clamped}%` }}
        />
      </div>

      {/* Value label */}
      <p className={clsx("mt-2 text-center text-lg font-bold", color)}>
        {clamped.toFixed(1)}th percentile
      </p>
    </div>
  );
}
