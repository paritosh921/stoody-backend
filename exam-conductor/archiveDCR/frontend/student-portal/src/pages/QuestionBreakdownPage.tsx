import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchScores } from "@/api/student-api";
import type { QuestionScore } from "@/types/api";
import clsx from "clsx";

function QuestionRow({
  q,
  examId,
  index,
}: {
  q: QuestionScore;
  examId: string;
  index: number;
}) {
  const pct = q.max_marks > 0 ? (q.marks_obtained / q.max_marks) * 100 : 0;

  return (
    <tr className="border-b border-gray-100 last:border-0">
      <td className="py-3 pr-4 text-sm font-medium text-gray-900">
        Q{index + 1}
      </td>
      <td className="py-3 pr-4 text-sm text-gray-700">
        {q.marks_obtained} / {q.max_marks}
      </td>
      <td className="py-3 pr-4">
        <div className="flex items-center gap-2">
          <div className="h-2 w-24 overflow-hidden rounded-full bg-gray-200">
            <div
              className={clsx(
                "h-full rounded-full",
                pct >= 70
                  ? "bg-green-500"
                  : pct >= 40
                    ? "bg-yellow-500"
                    : "bg-red-500",
              )}
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className="text-xs text-gray-500">{pct.toFixed(0)}%</span>
        </div>
      </td>
      <td className="py-3 pr-4 text-sm text-gray-500">
        {q.miss_indicator ?? "--"}
      </td>
      <td className="py-3 text-right">
        <Link
          to={`/scores/${examId}/answers/${q.question_id}`}
          className="text-sm font-medium text-primary-600 hover:text-primary-800"
        >
          View
        </Link>
      </td>
    </tr>
  );
}

export default function QuestionBreakdownPage() {
  const { examId } = useParams<{ examId: string }>();

  const { data, isLoading, error } = useQuery({
    queryKey: ["scores", examId],
    queryFn: () => fetchScores(examId!),
    enabled: !!examId,
  });

  if (isLoading) {
    return <p className="text-sm text-gray-400">Loading questions...</p>;
  }
  if (error || !data) {
    return (
      <p className="text-sm text-red-500">
        Failed to load question breakdown.
      </p>
    );
  }

  return (
    <div>
      <div className="flex items-center gap-2">
        <Link
          to={`/scores/${examId}`}
          className="text-sm text-primary-600 hover:text-primary-800"
        >
          &larr; Summary
        </Link>
        <h1 className="text-xl font-bold text-gray-900">
          Question Breakdown
        </h1>
      </div>

      <div className="mt-6 overflow-hidden rounded-lg border border-gray-200 bg-white">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-gray-200 bg-gray-50 text-xs font-medium uppercase text-gray-500">
            <tr>
              <th className="px-4 py-2">#</th>
              <th className="px-4 py-2">Score</th>
              <th className="px-4 py-2">Progress</th>
              <th className="px-4 py-2">Indicator</th>
              <th className="px-4 py-2 text-right">Answer</th>
            </tr>
          </thead>
          <tbody className="px-4">
            {data.questions.map((q, i) => (
              <QuestionRow
                key={q.question_id}
                q={q}
                examId={examId!}
                index={i}
              />
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4">
        <Link
          to={`/objections/file?examId=${examId}`}
          className="text-sm font-medium text-orange-600 hover:text-orange-800"
        >
          File an objection for this exam &rarr;
        </Link>
      </div>
    </div>
  );
}
