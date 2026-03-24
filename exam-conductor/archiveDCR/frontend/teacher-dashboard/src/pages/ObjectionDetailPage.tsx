// ---------------------------------------------------------------------------
// ObjectionDetailPage — view/resolve a single student objection + chat.
// ---------------------------------------------------------------------------

import { useState, type FormEvent } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useParams, useNavigate } from 'react-router-dom';
import { getObjectionDetail, resolveObjection, escalateObjection } from '@/api/teacher-api';
import { StatusBadge } from '@/components/StatusBadge';
import { ChatPanel } from '@/components/ChatPanel';

export function ObjectionDetailPage() {
  const { objectionId } = useParams<{ objectionId: string }>();
  const navigate = useNavigate();
  const qc = useQueryClient();

  const { data: objection, isLoading } = useQuery({
    queryKey: ['objection-detail', objectionId],
    queryFn: () => getObjectionDetail(objectionId!),
    enabled: !!objectionId,
    select: (res) => res.data,
  });

  const [verdict, setVerdict] = useState('accepted');
  const [comment, setComment] = useState('');

  const resolve = useMutation({
    mutationFn: () =>
      resolveObjection(objectionId!, { verdict, reason: comment }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['objections'] });
      qc.invalidateQueries({ queryKey: ['objection-detail', objectionId] });
      navigate(-1);
    },
  });

  const escalate = useMutation({
    mutationFn: () =>
      escalateObjection(objectionId!, { target_role: 'hod', reason: comment }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['objection-detail', objectionId] });
    },
  });

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!comment.trim()) return;
    resolve.mutate();
  }

  if (isLoading) return <Shell>Loading...</Shell>;
  if (!objection) return <Shell>Objection not found.</Shell>;

  return (
    <Shell>
      <button onClick={() => navigate(-1)}
        className="text-sm text-brand-600 hover:underline">
        &larr; Back to Inbox
      </button>

      <div className="mt-4 rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
        <div className="flex items-start justify-between">
          <h1 className="text-lg font-semibold text-gray-900">
            Objection {objection.objection_id.slice(0, 8)}
          </h1>
          <StatusBadge status={objection.status} />
        </div>

        <dl className="mt-4 grid grid-cols-2 gap-4 text-sm">
          <div>
            <dt className="text-gray-500">Student</dt>
            <dd className="text-gray-900">{objection.student_id}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Question</dt>
            <dd className="text-gray-900">{objection.question_id}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Filed</dt>
            <dd className="text-gray-900">
              {new Date(objection.filed_at).toLocaleString()}
            </dd>
          </div>
        </dl>
      </div>

      {/* Resolution form */}
      {objection.status !== 'resolved' && (
        <form onSubmit={handleSubmit} className="mt-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Verdict</label>
            <select value={verdict} onChange={(e) => setVerdict(e.target.value)}
              className="mt-1 rounded-md border border-gray-300 px-3 py-2 text-sm">
              <option value="accepted">Accept</option>
              <option value="rejected">Reject</option>
              <option value="partial">Partial</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Comment <span className="text-red-500">*</span>
            </label>
            <textarea rows={3} value={comment}
              onChange={(e) => setComment(e.target.value)}
              className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm"
              placeholder="Explain the resolution..." />
          </div>

          <div className="flex gap-3">
            <button type="submit"
              disabled={!comment.trim() || resolve.isPending}
              className="rounded-md bg-brand-600 px-5 py-2 text-sm font-medium text-white
                         hover:bg-brand-700 disabled:opacity-50">
              {resolve.isPending ? 'Submitting...' : 'Resolve Objection'}
            </button>
            <button type="button"
              onClick={() => { if (comment.trim()) escalate.mutate(); }}
              disabled={!comment.trim() || escalate.isPending}
              className="rounded-md border border-gray-300 px-5 py-2 text-sm font-medium
                         text-gray-700 hover:bg-gray-50 disabled:opacity-50">
              {escalate.isPending ? 'Escalating...' : 'Escalate to HOD'}
            </button>
          </div>
        </form>
      )}

      {/* Chat thread — calls svc-chat directly */}
      <div className="mt-6">
        <ChatPanel examId={objection.exam_id}
          studentId={objection.student_id} />
      </div>
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto max-w-3xl">{children}</div>;
}
