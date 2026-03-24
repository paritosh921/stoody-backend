interface TrendPoint {
  label: string;
  value: number;
}

interface TrendChartProps {
  data: TrendPoint[];
}

/**
 * Simple SVG line chart showing score trend over exams.
 * No external charting library needed -- pure SVG.
 */
export default function TrendChart({ data }: TrendChartProps) {
  if (data.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-gray-400">
        No exam data to display.
      </p>
    );
  }

  const W = 600;
  const H = 200;
  const PAD_X = 40;
  const PAD_Y = 20;

  const maxVal = Math.max(...data.map((d) => d.value), 1);
  const minVal = Math.min(...data.map((d) => d.value), 0);
  const range = maxVal - minVal || 1;

  const stepX =
    data.length > 1 ? (W - PAD_X * 2) / (data.length - 1) : 0;

  function toX(i: number): number {
    return PAD_X + i * stepX;
  }

  function toY(v: number): number {
    return H - PAD_Y - ((v - minVal) / range) * (H - PAD_Y * 2);
  }

  const polyline = data
    .map((d, i) => `${toX(i)},${toY(d.value)}`)
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full"
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Grid lines */}
      {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
        const y = PAD_Y + frac * (H - PAD_Y * 2);
        const val = maxVal - frac * range;
        return (
          <g key={frac}>
            <line
              x1={PAD_X}
              y1={y}
              x2={W - PAD_X}
              y2={y}
              stroke="#e5e7eb"
              strokeWidth={1}
            />
            <text
              x={PAD_X - 6}
              y={y + 4}
              textAnchor="end"
              className="fill-gray-400 text-[10px]"
            >
              {val.toFixed(0)}
            </text>
          </g>
        );
      })}

      {/* Line */}
      <polyline
        points={polyline}
        fill="none"
        stroke="#3b82f6"
        strokeWidth={2.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />

      {/* Dots + labels */}
      {data.map((d, i) => (
        <g key={i}>
          <circle
            cx={toX(i)}
            cy={toY(d.value)}
            r={4}
            className="fill-primary-600 stroke-white"
            strokeWidth={2}
          />
          <text
            x={toX(i)}
            y={H - 4}
            textAnchor="middle"
            className="fill-gray-500 text-[9px]"
          >
            {d.label}
          </text>
        </g>
      ))}
    </svg>
  );
}
