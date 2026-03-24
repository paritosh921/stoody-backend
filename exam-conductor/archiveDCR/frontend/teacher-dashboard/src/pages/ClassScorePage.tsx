// ---------------------------------------------------------------------------
// ClassScorePage — class score overview table for an exam.
// ---------------------------------------------------------------------------

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useParams, useNavigate } from 'react-router-dom';
import {
  getClassScores,
  finalizeScores,
  publishScores,
  type ClassScoreRow,
} from '@/api/teacher-api';
import { ScoreTable, type Column } from '@/components/ScoreTable';
import { ConfidenceBar } from '@/components/ConfidenceBar';
import { StatusBadge } from '@/components/StatusBadge';

export function ClassScorePage() {
  const { examId } = useParams<{ examId: string }>();
  const navigate = useNavigate();
  const qc = useQueryClient();

  const { data: rows, isLoading, error } = useQuery({
    queryKey: ['class-scores', examId],
    queryFn: () => getClassScores(examId!),
    enabled: !!examId,
    select: (res) => res.data?.rows ?? [],
  });

  const finalize = useMutation({
    mutationFn: () => finalizeScores(examId!),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['class-scores', examId] }),
  });

  const publish = useMutation({
    mutationFn: () => publishScores(examId!),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['class-scores', examId] }),
  });

  const columns: Column<ClassScoreRow>[] = [
    {
      key: 'student_name',
      header: 'Student',
      render: (r) => r.student_name,
      sortValue: (r) => r.student_name,
    },
    {
      key: 'total_score',
      header: 'Score',
      render: (r) => <span className="font-medium">{r.total_score}</span>,
      sortValue: (r) => r.total_score,
    },
    {
      key: 'percentile',
      header: 'Percentile',
      render: (r) => (r.percentile != null ? `${r.percentile}%` : '-'),
      sortValue: (r) => r.percentile ?? 0,
    },
    {
      key: 'ai_confidence',
      header: 'AI Confidence',
      render: (r) => <ConfidenceBar value={r.ai_confidence} className="w-32" />,
      sortValue: (r) => r.ai_confidence,
    },
    {
      key: 'flags',
      header: 'Flags',
      render: (r) => (
        <div className="flex gap-1">
          {(r.plagiarism_flag_count ?? 0) > 0 && (
            <StatusBadge status="high">Plagiarism</StatusBadge>
          )}
          {(r.miss_indicator_count ?? 0) > 0 && (
            <StatusBadge status="warning">Miss</StatusBadge>
          )}
        </div>
      ),
    },
  ];

  if (isLoading) return <Shell>Loading scores...</Shell>;
  if (error) return <Shell>Failed to load scores.</Shell>;

  return (
    <Shell>
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold text-gray-900">Class Scores</h1>
        <div className="flex gap-2">
          <button
            onClick={() => finalize.mutate()}
            disabled={finalize.isPending}
            className="rounded-md border border-gray-300 px-4 py-2 text-sm hover:bg-gray-50"
          >
            Finalize
          </button>
          <button
            onClick={() => publish.mutate()}
            disabled={publish.isPending}
            className="rounded-md bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700"
          >
            Publish
          </button>
        </div>
      </div>

      <ScoreTable
        columns={columns}
        data={rows ?? []}
        rowKey={(r) => r.student_id}
        searchAccessor={(r) => r.student_name}
        searchPlaceholder="Search students..."
        onRowClick={(r) => navigate(`/scores/${examId}/${r.student_id}`)}
      />
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto max-w-5xl">{children}</div>;
}
