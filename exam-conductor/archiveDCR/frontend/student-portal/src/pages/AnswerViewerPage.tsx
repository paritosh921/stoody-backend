import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchAnswerInsight } from "@/api/student-api";
import AnswerImage from "@/components/AnswerImage";

export default function AnswerViewerPage() {
  const { examId, questionId } = useParams<{
    examId: string;
    questionId: string;
  }>();

  const { data, isLoading, error } = useQuery({
    queryKey: ["answer", examId, questionId],
    queryFn: () => fetchAnswerInsight(examId!, questionId!),
    enabled: !!examId && !!questionId,
  });

  if (isLoading) {
    return <p className="text-sm text-gray-400">Loading answer...</p>;
  }
  if (error || !data) {
    return (
      <p className="text-sm text-red-500">Failed to load answer details.</p>
    );
  }

  return (
    <div>
      <div className="flex items-center gap-2">
        <Link
          to={`/scores/${examId}/questions`}
          className="text-sm text-primary-600 hover:text-primary-800"
        >
          &larr; Questions
        </Link>
        <h1 className="text-xl font-bold text-gray-900">Answer Viewer</h1>
      </div>

      {/* Answer image */}
      <div className="mt-6 rounded-lg border border-gray-200 bg-white p-4">
        <AnswerImage src={data.answer_image_uri} alt="Student answer" />
      </div>

      {/* Recognized text */}
      <div className="mt-4 rounded-lg border border-gray-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-gray-700">
          Recognized Text
        </h2>
        <p className="mt-2 whitespace-pre-wrap text-sm text-gray-600">
          {data.recognized_text}
        </p>
        <p className="mt-2 text-xs text-gray-400">
          Confidence: {(data.confidence * 100).toFixed(1)}%
        </p>
      </div>

      {/* Step breakdown */}
      {data.step_breakdown && data.step_breakdown.length > 0 && (
        <div className="mt-4 rounded-lg border border-gray-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-gray-700">
            Step Breakdown
          </h2>
          <ol className="mt-2 list-inside list-decimal space-y-1 text-sm text-gray-600">
            {data.step_breakdown.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ol>
        </div>
      )}

      {/* Feedback */}
      {data.feedback && (
        <div className="mt-4 rounded-lg border border-blue-100 bg-blue-50 p-5">
          <h2 className="text-sm font-semibold text-blue-800">Feedback</h2>
          <p className="mt-1 text-sm text-blue-700">{data.feedback}</p>
        </div>
      )}

      <div className="mt-4">
        <Link
          to={`/objections/file?examId=${examId}&questionId=${questionId}`}
          className="text-sm font-medium text-orange-600 hover:text-orange-800"
        >
          File objection for this question &rarr;
        </Link>
      </div>
    </div>
  );
}
