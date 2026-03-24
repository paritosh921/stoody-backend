import { useQuery } from "@tanstack/react-query";
import { fetchObjections } from "@/api/student-api";
import { Link } from "react-router-dom";
import type { StudentObjection, ObjectionStatus } from "@/types/api";
import clsx from "clsx";

const STATUS_STYLES: Record<ObjectionStatus, string> = {
  filed: "bg-blue-100 text-blue-700",
  assigned: "bg-indigo-100 text-indigo-700",
  reviewing: "bg-yellow-100 text-yellow-700",
  resolved: "bg-green-100 text-green-700",
  escalated: "bg-red-100 text-red-700",
};

function ObjectionCard({ obj }: { obj: StudentObjection }) {
  const colorCls = STATUS_STYLES[obj.status];

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-gray-900">
            Exam: {obj.exam_id.slice(0, 8)}...
          </p>
          <p className="text-xs text-gray-500">
            Question: {obj.question_id.slice(0, 8)}...
          </p>
        </div>
        <span
          className={clsx(
            "inline-flex rounded-full px-2.5 py-0.5 text-xs font-medium",
            colorCls,
          )}
        >
          {obj.status}
        </span>
      </div>

      {obj.objection_text && (
        <p className="mt-3 text-sm text-gray-600">{obj.objection_text}</p>
      )}

      {obj.resolution_reason && (
        <div className="mt-3 rounded-md bg-green-50 p-3">
          <p className="text-xs font-medium text-green-800">Resolution</p>
          <p className="text-sm text-green-700">{obj.resolution_reason}</p>
          {obj.new_score !== undefined && (
            <p className="mt-1 text-xs text-green-600">
              New score: {obj.new_score}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default function ObjectionStatusPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["objections"],
    queryFn: fetchObjections,
  });

  const objections = data?.items ?? [];

  return (
    <div>
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-900">Objections</h1>
          <p className="mt-1 text-sm text-gray-500">
            Track the status of your filed objections.
          </p>
        </div>
        <Link
          to="/objections/file"
          className="rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700"
        >
          File New
        </Link>
      </div>

      {isLoading && (
        <p className="mt-8 text-sm text-gray-400">Loading objections...</p>
      )}
      {error && (
        <p className="mt-8 text-sm text-red-500">
          Failed to load objections.
        </p>
      )}

      {!isLoading && objections.length === 0 && (
        <p className="mt-8 text-sm text-gray-400">
          You have not filed any objections yet.
        </p>
      )}

      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        {objections.map((obj) => (
          <ObjectionCard key={obj.objection_id} obj={obj} />
        ))}
      </div>
    </div>
  );
}
