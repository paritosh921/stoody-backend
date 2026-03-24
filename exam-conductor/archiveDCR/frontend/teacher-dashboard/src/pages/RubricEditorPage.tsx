// ---------------------------------------------------------------------------
// RubricEditorPage — per-question rubric editor with step breakdowns.
// ---------------------------------------------------------------------------

import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { getExamDetail, getRubric, saveRubric } from '@/api/teacher-api';
import { RubricQuestionRow, type RubricQuestion } from '@/components/RubricQuestionRow';

const ANSWER_TYPES = ['text', 'formula', 'diagram'] as const;
const DEFAULT_CONFIDENCE = 0.85;

export function RubricEditorPage() {
  const { examId } = useParams<{ examId: string }>();

  const { data: exam } = useQuery({
    queryKey: ['exam', examId],
    queryFn: () => getExamDetail(examId!),
    enabled: !!examId,
    select: (res) => res.data,
  });

  const { data: existingRubric, isLoading } = useQuery({
    queryKey: ['rubric', examId],
    queryFn: () => getRubric(examId!),
    enabled: !!examId,
    select: (res) => res.data,
  });

  const [questions, setQuestions] = useState<RubricQuestion[]>([]);
  const [confidence, setConfidence] = useState(DEFAULT_CONFIDENCE);
  const [initialized, setInitialized] = useState(false);

  // Seed state from loaded rubric (once)
  if (existingRubric && !initialized) {
    setQuestions(existingRubric.questions ?? buildEmptyQuestions(10));
    setConfidence(existingRubric.confidence_threshold ?? DEFAULT_CONFIDENCE);
    setInitialized(true);
  } else if (!isLoading && !existingRubric && !initialized) {
    setQuestions(buildEmptyQuestions(10));
    setInitialized(true);
  }

  const save = useMutation({
    mutationFn: () =>
      saveRubric(examId!, { questions, confidence_threshold: confidence }),
  });

  function updateQuestion(idx: number, q: RubricQuestion) {
    setQuestions((prev) => prev.map((old, i) => (i === idx ? q : old)));
  }

  function addQuestion() {
    const num = questions.length + 1;
    setQuestions((prev) => [
      ...prev,
      { question_number: num, max_marks: 10, answer_type: 'text', steps: [] },
    ]);
  }

  function removeQuestion(idx: number) {
    setQuestions((prev) => prev.filter((_, i) => i !== idx));
  }

  const stepsValid = questions.every((q) => {
    if (q.steps.length === 0) return true;
    const sum = q.steps.reduce((s, st) => s + st.marks, 0);
    return sum === q.max_marks;
  });

  return (
    <div className="mx-auto max-w-4xl">
      <Link to={`/exams/${examId}`} className="text-sm text-brand-600 hover:underline">
        &larr; Back to Exam
      </Link>

      <h1 className="mt-4 text-xl font-semibold text-gray-900">Rubric Editor</h1>
      <p className="mt-1 text-sm text-gray-500">
        {exam?.title ?? `Exam ${examId}`}
      </p>

      {isLoading ? (
        <p className="mt-6 text-gray-400">Loading rubric...</p>
      ) : (
        <>
          {/* Confidence threshold slider */}
          <div className="mt-6 rounded-lg border border-gray-200 bg-white p-4">
            <label className="block text-sm font-medium text-gray-700">
              AI Confidence Threshold: {confidence.toFixed(2)}
            </label>
            <input type="range" min={0.5} max={1.0} step={0.01} value={confidence}
              onChange={(e) => setConfidence(+e.target.value)}
              className="mt-2 w-full accent-brand-600" />
            <p className="mt-1 text-xs text-gray-400">
              Scores below this confidence will be flagged for manual review.
            </p>
          </div>

          {/* Question rows */}
          <div className="mt-4 space-y-3">
            {questions.map((q, idx) => (
              <RubricQuestionRow key={idx} question={q}
                answerTypes={ANSWER_TYPES}
                onChange={(updated) => updateQuestion(idx, updated)}
                onRemove={() => removeQuestion(idx)} />
            ))}
          </div>

          <button onClick={addQuestion} type="button"
            className="mt-3 text-sm font-medium text-brand-600 hover:underline">
            + Add Question
          </button>

          {!stepsValid && (
            <p className="mt-2 text-sm text-red-600">
              Step marks must sum to the question max marks.
            </p>
          )}

          <div className="mt-6 flex gap-3">
            <button onClick={() => save.mutate()} disabled={!stepsValid || save.isPending}
              className="rounded-md bg-brand-600 px-5 py-2 text-sm font-medium text-white
                         hover:bg-brand-700 disabled:opacity-50">
              {save.isPending ? 'Saving...' : 'Save Rubric'}
            </button>
            {save.isSuccess && (
              <span className="self-center text-sm text-green-600">Saved!</span>
            )}
            {save.isError && (
              <span className="self-center text-sm text-red-600">Save failed.</span>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function buildEmptyQuestions(count: number): RubricQuestion[] {
  return Array.from({ length: count }, (_, i) => ({
    question_number: i + 1,
    max_marks: 10,
    answer_type: 'text' as const,
    steps: [],
  }));
}
