import { useQuery } from "@tanstack/react-query";
import { fetchPerformance } from "@/api/student-api";
import TrendChart from "@/components/TrendChart";
import PercentileChart from "@/components/PercentileChart";

export default function PerformancePage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["performance"],
    queryFn: fetchPerformance,
  });

  if (isLoading) {
    return <p className="text-sm text-gray-400">Loading performance...</p>;
  }
  if (error || !data) {
    return (
      <p className="text-sm text-red-500">
        Failed to load performance data.
      </p>
    );
  }

  const latestPercentile =
    data.history.length > 0
      ? data.history[data.history.length - 1]!.percentile
      : 0;

  return (
    <div>
      <h1 className="text-xl font-bold text-gray-900">Performance</h1>
      <p className="mt-1 text-sm text-gray-500">
        Your score trends, percentile, and areas for improvement.
      </p>

      {/* Trend chart */}
      <div className="mt-6 rounded-lg border border-gray-200 bg-white p-5">
        <h2 className="mb-3 text-sm font-semibold text-gray-700">
          Score Trend
        </h2>
        <TrendChart
          data={data.history.map((h) => ({
            label: h.exam_id.slice(0, 6),
            value: h.score,
          }))}
        />
      </div>

      {/* Latest percentile */}
      <div className="mt-4 rounded-lg border border-gray-200 bg-white p-5">
        <h2 className="mb-3 text-sm font-semibold text-gray-700">
          Latest Percentile
        </h2>
        <PercentileChart percentile={latestPercentile} />
      </div>

      {/* Strengths & Weaknesses */}
      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <div className="rounded-lg border border-green-200 bg-green-50 p-5">
          <h2 className="text-sm font-semibold text-green-800">Strengths</h2>
          {data.strengths.length > 0 ? (
            <ul className="mt-2 list-inside list-disc space-y-1 text-sm text-green-700">
              {data.strengths.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-sm text-green-600">
              Not enough data yet.
            </p>
          )}
        </div>

        <div className="rounded-lg border border-red-200 bg-red-50 p-5">
          <h2 className="text-sm font-semibold text-red-800">
            Areas for Improvement
          </h2>
          {data.weaknesses.length > 0 ? (
            <ul className="mt-2 list-inside list-disc space-y-1 text-sm text-red-700">
              {data.weaknesses.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-sm text-red-600">
              Not enough data yet.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
