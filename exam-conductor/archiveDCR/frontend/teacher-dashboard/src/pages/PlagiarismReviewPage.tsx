// ---------------------------------------------------------------------------
// PlagiarismReviewPage — review plagiarism flags for an exam.
// ---------------------------------------------------------------------------

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useParams, Link } from 'react-router-dom';
import {
  getPlagiarismFlags,
  submitVerdict,
  type PlagiarismPreview,
} from '@/api/teacher-api';
import { StatusBadge } from '@/components/StatusBadge';
import { ConfidenceBar } from '@/components/ConfidenceBar';

export function PlagiarismReviewPage() {
  const { examId } = useParams<{ examId: string }>();
  const qc = useQueryClient();

  const { data: flags, isLoading, error } = useQuery({
    queryKey: ['plagiarism', examId],
    queryFn: () => getPlagiarismFlags(examId!),
    enabled: !!examId,
    select: (res) => res.data?.items ?? [],
  });

  const [active, setActive] = useState<PlagiarismPreview | null>(null);
  const [verdict, setVerdict] = useState('confirmed');
  const [notes, setNotes] = useState('');

  const submit = useMutation({
    mutationFn: () =>
      submitVerdict(active!.flag_id, { verdict, reason: notes }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['plagiarism', examId] });
      setActive(null);
      setNotes('');
    },
  });

  if (isLoading) return <Shell>Loading plagiarism flags...</Shell>;
  if (error) return <Shell>Failed to load plagiarism data.</Shell>;

  return (
    <Shell>
      <Link to="/exams" className="text-sm text-brand-600 hover:underline">
        &larr; Exams
      </Link>
      <h1 className="mt-4 text-xl font-semibold text-gray-900">
        Plagiarism Review
      </h1>

      {flags && flags.length === 0 && (
        <p className="mt-4 text-gray-500">No plagiarism flags for this exam.</p>
      )}

      <div className="mt-4 space-y-3">
        {flags?.map((f) => (
          <div
            key={f.flag_id}
            className="rounded-lg border border-gray-200 bg-white px-4 py-3"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-900">
                  {f.student_a_id} &harr; {f.student_b_id}
                </p>
                <p className="text-xs text-gray-500">Q: {f.question_id}</p>
              </div>
              <div className="flex items-center gap-3">
                <ConfidenceBar value={f.composite_score} className="w-24" />
                <StatusBadge status={f.severity} />
                {f.teacher_verdict ? (
                  <StatusBadge status={f.teacher_verdict} variant="info" />
                ) : (
                  <button
                    onClick={() => setActive(f)}
                    className="rounded border border-gray-300 px-2 py-1 text-xs hover:bg-gray-50"
                  >
                    Review
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Verdict modal */}
      {active && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="w-full max-w-md rounded-lg bg-white p-6 shadow-xl">
            <h3 className="text-lg font-semibold">Submit Verdict</h3>
            <p className="mt-1 text-sm text-gray-500">
              {active.student_a_id} &harr; {active.student_b_id}
            </p>

            <div className="mt-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Verdict
                </label>
                <select
                  value={verdict}
                  onChange={(e) => setVerdict(e.target.value)}
                  className="mt-1 rounded-md border border-gray-300 px-3 py-2 text-sm"
                >
                  <option value="confirmed">Confirmed plagiarism</option>
                  <option value="dismissed">Dismissed</option>
                  <option value="inconclusive">Inconclusive</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Notes
                </label>
                <textarea
                  rows={2}
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm"
                />
              </div>
              <div className="flex justify-end gap-3">
                <button
                  onClick={() => setActive(null)}
                  className="rounded-md px-4 py-2 text-sm text-gray-600 hover:bg-gray-100"
                >
                  Cancel
                </button>
                <button
                  onClick={() => submit.mutate()}
                  disabled={submit.isPending}
                  className="rounded-md bg-brand-600 px-4 py-2 text-sm font-medium text-white
                             hover:bg-brand-700 disabled:opacity-50"
                >
                  {submit.isPending ? 'Saving...' : 'Submit'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto max-w-5xl">{children}</div>;
}
