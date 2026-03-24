// ---------------------------------------------------------------------------
// ObjectionInboxPage — list of student objections for the teacher.
// ---------------------------------------------------------------------------

import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { getObjections } from '@/api/teacher-api';
import { StatusBadge } from '@/components/StatusBadge';

export function ObjectionInboxPage() {
  const { examId } = useParams<{ examId: string }>();

  const { data: items, isLoading, error } = useQuery({
    queryKey: ['objections', examId],
    queryFn: () => getObjections(examId!),
    enabled: !!examId,
    select: (res) => res.data?.items ?? [],
  });

  if (isLoading) return <Shell>Loading objections...</Shell>;
  if (error) return <Shell>Failed to load objections.</Shell>;

  return (
    <Shell>
      <h1 className="mb-4 text-xl font-semibold text-gray-900">
        Objection Inbox
      </h1>

      {items && items.length === 0 && (
        <p className="text-gray-500">No objections filed.</p>
      )}

      <div className="space-y-3">
        {items?.map((obj) => (
          <Link
            key={obj.objection_id}
            to={`/objections/detail/${obj.objection_id}`}
            className="flex items-center justify-between rounded-lg border border-gray-200
                       bg-white px-4 py-3 transition hover:shadow-sm"
          >
            <div>
              <p className="text-sm font-medium text-gray-900">
                Student: {obj.student_id}
              </p>
              <p className="text-xs text-gray-500">
                Question: {obj.question_id} &middot;{' '}
                {new Date(obj.filed_at).toLocaleDateString()}
              </p>
            </div>
            <StatusBadge status={obj.status} />
          </Link>
        ))}
      </div>
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto max-w-4xl">{children}</div>;
}
