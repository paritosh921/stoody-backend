import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createObjection } from "@/api/student-api";
import type { StudentExamCard, QuestionScore } from "@/types/api";

interface ObjectionFormProps {
  exams: StudentExamCard[];
  questions: QuestionScore[];
  defaultExamId: string;
  defaultQuestionId: string;
  onSuccess: () => void;
}

export default function ObjectionForm({
  exams,
  questions,
  defaultExamId,
  defaultQuestionId,
  onSuccess,
}: ObjectionFormProps) {
  const queryClient = useQueryClient();
  const [examId, setExamId] = useState(defaultExamId);
  const [questionId, setQuestionId] = useState(defaultQuestionId);
  const [text, setText] = useState("");

  const mutation = useMutation({
    mutationFn: createObjection,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["objections"] });
      onSuccess();
    },
  });

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!examId || !questionId || !text.trim()) return;
    mutation.mutate({
      exam_id: examId,
      question_id: questionId,
      objection_text: text.trim(),
    });
  }

  const canSubmit =
    !!examId && !!questionId && text.trim().length > 0 && !mutation.isPending;

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Exam selector */}
      <div>
        <label
          htmlFor="obj-exam"
          className="mb-1 block text-sm font-medium text-gray-700"
        >
          Exam
        </label>
        <select
          id="obj-exam"
          value={examId}
          onChange={(e) => {
            setExamId(e.target.value);
            setQuestionId("");
          }}
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
        >
          <option value="">Select exam...</option>
          {exams.map((ex) => (
            <option key={ex.exam_id} value={ex.exam_id}>
              {ex.title}
            </option>
          ))}
        </select>
      </div>

      {/* Question selector */}
      <div>
        <label
          htmlFor="obj-question"
          className="mb-1 block text-sm font-medium text-gray-700"
        >
          Question
        </label>
        <select
          id="obj-question"
          value={questionId}
          onChange={(e) => setQuestionId(e.target.value)}
          disabled={questions.length === 0}
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500 disabled:opacity-50"
        >
          <option value="">
            {questions.length === 0
              ? "Select an exam first"
              : "Select question..."}
          </option>
          {questions.map((q, i) => (
            <option key={q.question_id} value={q.question_id}>
              Q{i + 1} ({q.marks_obtained}/{q.max_marks})
            </option>
          ))}
        </select>
      </div>

      {/* Objection text */}
      <div>
        <label
          htmlFor="obj-text"
          className="mb-1 block text-sm font-medium text-gray-700"
        >
          Objection Details
        </label>
        <textarea
          id="obj-text"
          rows={4}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Explain why you believe the score is incorrect..."
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
        />
      </div>

      {/* Error message */}
      {mutation.isError && (
        <p className="text-sm text-red-500">
          Failed to submit objection. Please try again.
        </p>
      )}

      <button
        type="submit"
        disabled={!canSubmit}
        className="w-full rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 disabled:opacity-50"
      >
        {mutation.isPending ? "Submitting..." : "Submit Objection"}
      </button>
    </form>
  );
}
