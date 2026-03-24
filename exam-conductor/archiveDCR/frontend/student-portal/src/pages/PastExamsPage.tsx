import { useQuery } from "@tanstack/react-query";
import { fetchExams } from "@/api/student-api";
import type { StudentExamCard } from "@/types/api";
import { Link } from "react-router-dom";
import clsx from "clsx";

const STATUS_COLORS: Record<string, string> = {
  published: "bg-green-100 text-green-700",
  scores_pending: "bg-yellow-100 text-yellow-700",
  objection_window_open: "bg-orange-100 text-orange-700",
  locked: "bg-gray-100 text-gray-600",
};

function PastExamRow({ exam }: { exam: StudentExamCard }) {
  const date = new Date(exam.scheduled_at);
  const colorCls = STATUS_COLORS[exam.status] ?? "bg-gray-100 text-gray-600";

  return (
    <tr className="border-b border-gray-100 last:border-0">
      <td className="py-3 pr-4">
        <p className="font-medium text-gray-900">{exam.title}</p>
        {exam.subject_name && (
          <p className="text-xs text-gray-500">{exam.subject_name}</p>
        )}
      </td>
      <td className="py-3 pr-4 text-sm text-gray-600">
        {date.toLocaleDateString(undefined, {
          month: "short",
          day: "numeric",
          year: "numeric",
        })}
      </td>
      <td className="py-3 pr-4">
        <span
          className={clsx(
            "inline-flex rounded-full px-2 py-0.5 text-xs font-medium",
            colorCls,
          )}
        >
          {exam.status.replace(/_/g, " ")}
        </span>
      </td>
      <td className="py-3 text-right">
        {exam.status !== "scores_pending" ? (
          <Link
            to={`/scores/${exam.exam_id}`}
            className="text-sm font-medium text-primary-600 hover:text-primary-800"
          >
            Scores
          </Link>
        ) : (
          <span className="text-sm text-gray-400">Pending</span>
        )}
      </td>
    </tr>
  );
}

export default function PastExamsPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["exams"],
    queryFn: fetchExams,
  });

  const past = data?.items.filter((e) => e.status !== "upcoming") ?? [];

  return (
    <div>
      <h1 className="text-xl font-bold text-gray-900">Past Exams</h1>
      <p className="mt-1 text-sm text-gray-500">
        Completed exams and their results.
      </p>

      {isLoading && (
        <p className="mt-8 text-sm text-gray-400">Loading...</p>
      )}
      {error && (
        <p className="mt-8 text-sm text-red-500">
          Failed to load exams. Please try again.
        </p>
      )}

      {!isLoading && past.length === 0 && (
        <p className="mt-8 text-sm text-gray-400">No past exams yet.</p>
      )}

      {past.length > 0 && (
        <div className="mt-6 overflow-hidden rounded-lg border border-gray-200 bg-white">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-gray-200 bg-gray-50 text-xs font-medium uppercase text-gray-500">
              <tr>
                <th className="px-4 py-2">Exam</th>
                <th className="px-4 py-2">Date</th>
                <th className="px-4 py-2">Status</th>
                <th className="px-4 py-2 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="px-4">
              {past.map((exam) => (
                <PastExamRow key={exam.exam_id} exam={exam} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
