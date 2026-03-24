import { useSearchParams, Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchExams, fetchScores } from "@/api/student-api";
import ObjectionForm from "@/components/ObjectionForm";

export default function ObjectionFilePage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const preExamId = searchParams.get("examId") ?? "";
  const preQuestionId = searchParams.get("questionId") ?? "";

  // Fetch exam list for the selector
  const { data: examsData } = useQuery({
    queryKey: ["exams"],
    queryFn: fetchExams,
  });

  // Fetch questions if an exam is pre-selected
  const { data: scoresData } = useQuery({
    queryKey: ["scores", preExamId],
    queryFn: () => fetchScores(preExamId),
    enabled: !!preExamId,
  });

  const eligibleExams =
    examsData?.items.filter(
      (e) =>
        e.status === "objection_window_open" || e.status === "published",
    ) ?? [];

  const questions = scoresData?.questions ?? [];

  return (
    <div>
      <div className="flex items-center gap-2">
        <Link
          to="/objections/status"
          className="text-sm text-primary-600 hover:text-primary-800"
        >
          &larr; Objections
        </Link>
        <h1 className="text-xl font-bold text-gray-900">File an Objection</h1>
      </div>

      <div className="mt-6 max-w-xl rounded-lg border border-gray-200 bg-white p-6">
        <ObjectionForm
          exams={eligibleExams}
          questions={questions}
          defaultExamId={preExamId}
          defaultQuestionId={preQuestionId}
          onSuccess={() => navigate("/objections/status")}
        />
      </div>
    </div>
  );
}
