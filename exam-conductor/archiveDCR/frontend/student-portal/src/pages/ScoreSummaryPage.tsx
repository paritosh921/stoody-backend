import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchScores } from "@/api/student-api";
import PercentileChart from "@/components/PercentileChart";
import clsx from "clsx";

export default function ScoreSummaryPage() {
  const { examId } = useParams<{ examId: string }>();

  const { data, isLoading, error } = useQuery({
    queryKey: ["scores", examId],
    queryFn: () => fetchScores(examId!),
    enabled: !!examId,
  });

  if (isLoading) {
    return <p className="text-sm text-gray-400">Loading scores...</p>;
  }
  if (error || !data) {
    return (
      <p className="text-sm text-red-500">
        Failed to load scores. Please try again.
      </p>
    );
  }

  const passFail = data.pass_fail;

  return (
    <div>
      <div className="flex items-center gap-2">
        <Link
          to="/exams/past"
          className="text-sm text-primary-600 hover:text-primary-800"
        >
          &larr; Back
        </Link>
        <h1 className="text-xl font-bold text-gray-900">Score Summary</h1>
      </div>

      {/* Score cards */}
      <div className="mt-6 grid gap-4 sm:grid-cols-3">
        <div className="rounded-lg border border-gray-200 bg-white p-5">
          <p className="text-xs font-medium uppercase text-gray-500">
            Total Score
          </p>
          <p className="mt-1 text-2xl font-bold text-gray-900">
            {data.total_score}
          </p>
        </div>

        <div className="rounded-lg border border-gray-200 bg-white p-5">
          <p className="text-xs font-medium uppercase text-gray-500">
            Percentage
          </p>
          <p className="mt-1 text-2xl font-bold text-gray-900">
            {data.percentage.toFixed(1)}%
          </p>
        </div>

        <div className="rounded-lg border border-gray-200 bg-white p-5">
          <p className="text-xs font-medium uppercase text-gray-500">
            Result
          </p>
          <p
            className={clsx(
              "mt-1 text-2xl font-bold",
              passFail === "pass" ? "text-green-600" : "text-red-600",
            )}
          >
            {passFail === "pass" ? "Pass" : passFail === "fail" ? "Fail" : "--"}
          </p>
        </div>
      </div>

      {/* Percentile */}
      <div className="mt-6 rounded-lg border border-gray-200 bg-white p-5">
        <p className="mb-3 text-sm font-medium text-gray-700">
          Percentile Rank
        </p>
        <PercentileChart percentile={data.percentile} />
      </div>

      {/* Question breakdown link */}
      <div className="mt-6 flex gap-3">
        <Link
          to={`/scores/${examId}/questions`}
          className="rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700"
        >
          Question Breakdown
        </Link>
        {/* Chat requires a teacherId — the link should include the exam's
            teacher. When exam detail data is available, replace "teacher"
            segment below with the actual teacher_id from the exam. */}
        <Link
          to={`/chat/${examId}/${data.teacher_id ?? ''}`}
          className="rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
        >
          Chat with Teacher
        </Link>
      </div>
    </div>
  );
}
