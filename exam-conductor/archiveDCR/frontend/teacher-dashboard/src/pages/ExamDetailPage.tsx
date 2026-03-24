// ---------------------------------------------------------------------------
// ExamDetailPage — single exam overview with management actions.
// ---------------------------------------------------------------------------

import { useState, type ChangeEvent } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import {
  getExamDetail,
  uploadQuestionPaper,
  assignStaff,
} from '@/api/teacher-api';
import { StatusBadge } from '@/components/StatusBadge';

export function ExamDetailPage() {
  const { examId } = useParams<{ examId: string }>();

  const { data: exam, isLoading, error } = useQuery({
    queryKey: ['exam', examId],
    queryFn: () => getExamDetail(examId!),
    enabled: !!examId,
    select: (res) => res.data,
  });

  if (isLoading) return <Shell>Loading exam...</Shell>;
  if (error || !exam) return <Shell>Failed to load exam.</Shell>;

  return (
    <Shell>
      <div className="mb-6">
        <Link to="/exams" className="text-sm text-brand-600 hover:underline">
          &larr; All Exams
        </Link>
      </div>

      <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-gray-900">{exam.title}</h1>
            <p className="mt-1 text-sm text-gray-500">
              {exam.class_label ?? exam.subject_id}
            </p>
          </div>
          <StatusBadge status={exam.state} />
        </div>

        <dl className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Subject" value={exam.subject_id} />
          <Stat label="Scheduled" value={new Date(exam.scheduled_at).toLocaleString()} />
          <Stat label="State" value={exam.state} />
          <Stat label="Class" value={exam.class_label ?? '-'} />
        </dl>
      </div>

      {/* Primary navigation cards */}
      <div className="mt-6 grid gap-4 sm:grid-cols-3">
        <ActionCard to={`/scores/${examId}`} title="Class Scores" desc="View and override student scores" />
        <ActionCard to={`/exams/${examId}/rubric`} title="Rubric" desc="Edit grading rubric" />
        <ActionCard to={`/plagiarism/${examId}`} title="Plagiarism" desc="Review similarity flags" />
        <ActionCard to={`/question-regions/${examId}`} title="Question Regions" desc="Define answer sheet regions" />
        <ActionCard to={`/objections/${examId}`} title="Objections" desc="Review student objections" />
        <ActionCard to={`/analytics`} title="Analytics" desc="Class performance analytics" />
      </div>

      {/* Management actions */}
      <h2 className="mt-8 text-lg font-medium text-gray-900">Exam Management</h2>
      <div className="mt-3 grid gap-4 sm:grid-cols-2">
        <QuestionPaperUpload examId={examId!} />
        <StaffAssignPanel examId={examId!} />
      </div>
    </Shell>
  );
}

/* --- Sub-components ---------------------------------------------------- */

function QuestionPaperUpload({ examId }: { examId: string }) {
  const [file, setFile] = useState<File | null>(null);
  const upload = useMutation({
    mutationFn: async () => {
      const res = await uploadQuestionPaper(examId, file!);
      if (res.status === 501) throw new Error('Not yet implemented');
      return res;
    },
  });

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <h3 className="text-sm font-medium text-gray-900">Upload Question Paper</h3>
      <p className="mt-1 text-xs text-gray-500">Requires S3 integration (coming soon)</p>
      <input type="file" accept="application/pdf,image/*"
        onChange={(e: ChangeEvent<HTMLInputElement>) => setFile(e.target.files?.[0] ?? null)}
        className="mt-2 block text-sm text-gray-500 file:mr-3 file:rounded-md file:border-0
          file:bg-brand-50 file:px-3 file:py-1.5 file:text-xs file:font-medium
          file:text-brand-700 hover:file:bg-brand-100" />
      <button onClick={() => upload.mutate()} disabled={!file || upload.isPending}
        className="mt-2 rounded-md bg-gray-400 px-4 py-1.5 text-xs font-medium text-white
                   cursor-not-allowed" title="Not yet implemented">
        Upload (unavailable)
      </button>
      {upload.isError && <p className="mt-1 text-xs text-amber-600">Upload not yet available.</p>}
    </div>
  );
}

function StaffAssignPanel({ examId }: { examId: string }) {
  const [invigIds, setInvigIds] = useState('');
  const [evalIds, setEvalIds] = useState('');
  const assign = useMutation({
    mutationFn: () => {
      const invig = invigIds.split(',').map((s) => s.trim()).filter(Boolean);
      const eval_ = evalIds.split(',').map((s) => s.trim()).filter(Boolean);
      return assignStaff(examId, invig, eval_);
    },
  });

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <h3 className="text-sm font-medium text-gray-900">Assign Staff</h3>
      <div className="mt-2 space-y-2">
        <input type="text" value={invigIds} onChange={(e) => setInvigIds(e.target.value)}
          placeholder="Invigilator IDs (comma-separated)"
          className="w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm" />
        <input type="text" value={evalIds} onChange={(e) => setEvalIds(e.target.value)}
          placeholder="Evaluator IDs (comma-separated)"
          className="w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm" />
      </div>
      <button onClick={() => assign.mutate()}
        disabled={(!invigIds.trim() && !evalIds.trim()) || assign.isPending}
        className="mt-2 rounded-md bg-brand-600 px-4 py-1.5 text-xs font-medium text-white
                   hover:bg-brand-700 disabled:opacity-50">
        {assign.isPending ? 'Assigning...' : 'Assign Staff'}
      </button>
      {assign.isSuccess && <p className="mt-1 text-xs text-green-600">Staff assigned!</p>}
      {assign.isError && <p className="mt-1 text-xs text-red-600">Assignment failed.</p>}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs font-medium text-gray-500">{label}</dt>
      <dd className="mt-0.5 text-sm text-gray-900">{value}</dd>
    </div>
  );
}

function ActionCard({ to, title, desc }: { to: string; title: string; desc: string }) {
  return (
    <Link to={to}
      className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm
                 transition hover:shadow-md">
      <h3 className="font-medium text-brand-600">{title}</h3>
      <p className="mt-1 text-sm text-gray-500">{desc}</p>
    </Link>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto max-w-5xl">{children}</div>;
}
