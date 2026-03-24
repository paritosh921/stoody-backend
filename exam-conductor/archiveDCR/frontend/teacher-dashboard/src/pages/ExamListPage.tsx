// ---------------------------------------------------------------------------
// ExamListPage — lists all exams for the authenticated teacher.
// ---------------------------------------------------------------------------

import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { getExams, type TeacherExamCard } from '@/api/teacher-api';
import { StatusBadge } from '@/components/StatusBadge';

export function ExamListPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['exams'],
    queryFn: () => getExams(),
    select: (res) => res.data?.items ?? [],
  });

  if (isLoading) return <PageShell>Loading exams...</PageShell>;
  if (error) return <PageShell>Failed to load exams.</PageShell>;

  const exams = data ?? [];

  return (
    <PageShell>
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold text-gray-900">Exams</h1>
        <Link
          to="/exams/create"
          className="rounded-md bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700"
        >
          Create Exam
        </Link>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {exams.map((exam) => (
          <ExamCard key={exam.exam_id} exam={exam} />
        ))}
        {exams.length === 0 && (
          <p className="text-gray-500">No exams found.</p>
        )}
      </div>
    </PageShell>
  );
}

function ExamCard({ exam }: { exam: TeacherExamCard }) {
  return (
    <Link
      to={`/exams/${exam.exam_id}`}
      className="block rounded-lg border border-gray-200 bg-white p-4 shadow-sm
                 transition hover:shadow-md"
    >
      <div className="flex items-start justify-between">
        <h3 className="font-medium text-gray-900">{exam.title}</h3>
        <StatusBadge status={exam.state} />
      </div>
      <p className="mt-1 text-sm text-gray-500">
        {exam.class_label ?? exam.subject_id}
      </p>
      <p className="mt-2 text-xs text-gray-400">
        {new Date(exam.scheduled_at).toLocaleString()}
      </p>
    </Link>
  );
}

function PageShell({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto max-w-5xl">{children}</div>;
}
