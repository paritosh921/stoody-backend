import { useQuery } from "@tanstack/react-query";
import { fetchExams } from "@/api/student-api";
import type { StudentExamCard } from "@/types/api";
import { Link } from "react-router-dom";

function ExamCard({ exam }: { exam: StudentExamCard }) {
  const date = new Date(exam.scheduled_at);
  const isUpcoming = exam.status === "upcoming";

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-base font-semibold text-gray-900">
            {exam.title}
          </h3>
          {exam.subject_name && (
            <p className="mt-0.5 text-sm text-gray-500">
              {exam.subject_name}
            </p>
          )}
        </div>
        <span className="inline-flex rounded-full bg-blue-100 px-2.5 py-0.5 text-xs font-medium text-blue-700">
          {exam.status.replace("_", " ")}
        </span>
      </div>

      <p className="mt-3 text-sm text-gray-600">
        {date.toLocaleDateString(undefined, {
          weekday: "short",
          month: "short",
          day: "numeric",
        })}{" "}
        at{" "}
        {date.toLocaleTimeString(undefined, {
          hour: "2-digit",
          minute: "2-digit",
        })}
      </p>

      {!isUpcoming && (
        <Link
          to={`/scores/${exam.exam_id}`}
          className="mt-3 inline-block text-sm font-medium text-primary-600 hover:text-primary-800"
        >
          View scores &rarr;
        </Link>
      )}
    </div>
  );
}

export default function UpcomingExamsPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["exams"],
    queryFn: fetchExams,
  });

  const upcoming =
    data?.items.filter((e) => e.status === "upcoming") ?? [];

  return (
    <div>
      <h1 className="text-xl font-bold text-gray-900">Upcoming Exams</h1>
      <p className="mt-1 text-sm text-gray-500">
        Exams you haven't taken yet.
      </p>

      {isLoading && (
        <p className="mt-8 text-sm text-gray-400">Loading exams...</p>
      )}
      {error && (
        <p className="mt-8 text-sm text-red-500">
          Failed to load exams. Please try again.
        </p>
      )}

      {!isLoading && upcoming.length === 0 && (
        <p className="mt-8 text-sm text-gray-400">
          No upcoming exams scheduled.
        </p>
      )}

      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {upcoming.map((exam) => (
          <ExamCard key={exam.exam_id} exam={exam} />
        ))}
      </div>
    </div>
  );
}
