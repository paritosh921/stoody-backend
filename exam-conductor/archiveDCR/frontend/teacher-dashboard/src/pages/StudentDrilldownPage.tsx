// ---------------------------------------------------------------------------
// StudentDrilldownPage — per-student question breakdown with score override.
// ---------------------------------------------------------------------------

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { getStudentDetail, overrideScore } from '@/api/teacher-api';
import { ConfidenceBar } from '@/components/ConfidenceBar';
import { ScoreEditModal } from '@/components/ScoreEditModal';

export function StudentDrilldownPage() {
  const { examId, studentId } = useParams<{
    examId: string;
    studentId: string;
  }>();
  const qc = useQueryClient();
  const [editQ, setEditQ] = useState<{
    question_id: string;
    current_score: number;
  } | null>(null);

  const { data: detail, isLoading, error } = useQuery({
    queryKey: ['student-detail', examId, studentId],
    queryFn: () => getStudentDetail(examId!, studentId!),
    enabled: !!examId && !!studentId,
    select: (res) => res.data,
  });

  const mutation = useMutation({
    mutationFn: (p: { question_id: string; new_score: number; reason: string }) =>
      overrideScore(examId!, studentId!, p),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['student-detail', examId, studentId] });
      setEditQ(null);
    },
  });

  if (isLoading) return <Shell>Loading student detail...</Shell>;
  if (error || !detail) return <Shell>Failed to load student detail.</Shell>;

  return (
    <Shell>
      <Link
        to={`/scores/${examId}`}
        className="text-sm text-brand-600 hover:underline"
      >
        &larr; Back to Class Scores
      </Link>

      <div className="mt-4 rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-xl font-semibold text-gray-900">
              {detail.student_name}
            </h1>
            <p className="text-sm text-gray-500">ID: {detail.student_id}</p>
          </div>
          <span className="text-2xl font-bold text-brand-600">
            {detail.total_score}
          </span>
        </div>
      </div>

      <h2 className="mt-6 text-lg font-medium text-gray-900">Questions</h2>

      <div className="mt-3 space-y-3">
        {detail.questions.map((q) => (
          <div
            key={q.question_id}
            className="flex items-center justify-between rounded-lg border border-gray-200 bg-white px-4 py-3"
          >
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium text-gray-900">
                {q.question_id}
              </p>
              {q.recognized_text && (
                <p className="mt-0.5 truncate text-xs text-gray-500">
                  {q.recognized_text}
                </p>
              )}
              <ConfidenceBar value={q.confidence} className="mt-1 w-40" />
            </div>
            <div className="flex items-center gap-3">
              <span className="text-lg font-semibold">{q.current_score}</span>
              <button
                onClick={() =>
                  setEditQ({
                    question_id: q.question_id,
                    current_score: q.current_score,
                  })
                }
                className="rounded border border-gray-300 px-2 py-1 text-xs hover:bg-gray-50"
              >
                Override
              </button>
            </div>
          </div>
        ))}
      </div>

      {editQ && (
        <ScoreEditModal
          questionId={editQ.question_id}
          currentScore={editQ.current_score}
          onSubmit={(qid, score, reason) =>
            mutation.mutate({ question_id: qid, new_score: score, reason })
          }
          onClose={() => setEditQ(null)}
          isSubmitting={mutation.isPending}
        />
      )}
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto max-w-4xl">{children}</div>;
}
