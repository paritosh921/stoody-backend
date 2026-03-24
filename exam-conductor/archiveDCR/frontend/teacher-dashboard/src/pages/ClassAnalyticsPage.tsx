// ---------------------------------------------------------------------------
// ClassAnalyticsPage — exam-level class analytics with export support.
// ---------------------------------------------------------------------------

import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { getExams, getClassStats, exportAnalyticsCsv, type TeacherExamCard, type ClassStats } from '@/api/teacher-api';
import { StatusBadge } from '@/components/StatusBadge';

export function ClassAnalyticsPage() {
  const { data: exams, isLoading } = useQuery({
    queryKey: ['exams'],
    queryFn: () => getExams(),
    select: (res) => res.data?.items ?? [],
  });

  if (isLoading) return <Shell>Loading analytics...</Shell>;

  const completed = exams?.filter((e) => e.state === 'completed') ?? [];

  return (
    <Shell>
      <h1 className="mb-4 text-xl font-semibold text-gray-900">Class Analytics</h1>

      {completed.length === 0 && (
        <p className="text-gray-500">No completed exams with analytics data.</p>
      )}

      <div className="space-y-4">
        {completed.map((exam) => (
          <ExamStatsCard key={exam.exam_id} exam={exam} />
        ))}
      </div>
    </Shell>
  );
}

function ExamStatsCard({ exam }: { exam: TeacherExamCard }) {
  const { data: stats } = useQuery({
    queryKey: ['class-stats', exam.exam_id],
    queryFn: () => getClassStats(exam.exam_id),
    select: (res) => res.data,
  });

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="font-medium text-gray-900">{exam.title}</h2>
          <p className="text-sm text-gray-500">
            {exam.class_label ?? exam.subject_id}
          </p>
        </div>
        <StatusBadge status={exam.state} />
      </div>

      {stats ? (
        <>
          <dl className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Mean" value={stats.mean.toFixed(1)} />
            <Stat label="Median" value={stats.median.toFixed(1)} />
            <Stat label="Std Dev" value={stats.std_dev.toFixed(2)} />
            <Stat label="Pass Rate" value={`${(stats.pass_rate * 100).toFixed(0)}%`} />
          </dl>

          <div className="mt-4 flex items-center gap-3">
            <Link to={`/analytics/${exam.exam_id}/leaderboard`}
              className="text-sm font-medium text-brand-600 hover:underline">
              View Leaderboard &rarr;
            </Link>
            <ExportButtons examId={exam.exam_id} stats={stats} />
          </div>
        </>
      ) : (
        <p className="mt-4 text-sm text-gray-400">Loading stats...</p>
      )}
    </div>
  );
}

function ExportButtons({ examId, stats }: { examId: string; stats: ClassStats }) {
  function handleExportCsv() {
    // Try server-side export first (BFF returns {format, content, stats})
    exportAnalyticsCsv(examId).then((res) => {
      const content = (res.data as any)?.content;
      if (content && typeof content === 'string') {
        downloadBlob(content, `analytics-${examId}.csv`, 'text/csv');
      } else {
        // Client-side fallback from loaded stats
        const csv = buildStatsCsv(stats);
        downloadBlob(csv, `analytics-${examId}.csv`, 'text/csv');
      }
    }).catch(() => {
      const csv = buildStatsCsv(stats);
      downloadBlob(csv, `analytics-${examId}.csv`, 'text/csv');
    });
  }

  function handleExportPdf() {
    // Client-side PDF generation using a printable view
    const content = [
      `Class Analytics Report - Exam ${examId}`,
      '',
      `Mean: ${stats.mean.toFixed(1)}`,
      `Median: ${stats.median.toFixed(1)}`,
      `Std Dev: ${stats.std_dev.toFixed(2)}`,
      `Pass Rate: ${(stats.pass_rate * 100).toFixed(0)}%`,
    ].join('\n');

    const printWindow = window.open('', '_blank');
    if (printWindow) {
      printWindow.document.write(
        `<html><head><title>Analytics Report</title></head><body><pre>${content}</pre></body></html>`,
      );
      printWindow.document.close();
      printWindow.print();
    }
  }

  return (
    <div className="flex gap-2 ml-auto">
      <button onClick={handleExportCsv}
        className="rounded-md border border-gray-300 px-3 py-1 text-xs font-medium
                   text-gray-700 hover:bg-gray-50">
        Export CSV
      </button>
      <button onClick={handleExportPdf}
        className="rounded-md border border-gray-300 px-3 py-1 text-xs font-medium
                   text-gray-700 hover:bg-gray-50">
        Export PDF
      </button>
    </div>
  );
}

function buildStatsCsv(stats: ClassStats): string {
  const header = 'Metric,Value';
  const rows = [
    `Mean,${stats.mean}`,
    `Median,${stats.median}`,
    `Std Dev,${stats.std_dev}`,
    `Pass Rate,${stats.pass_rate}`,
  ];
  return [header, ...rows].join('\n');
}

function downloadBlob(content: string | Blob, filename: string, type: string) {
  const blob = content instanceof Blob ? content : new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-gray-500">{label}</dt>
      <dd className="text-lg font-semibold text-gray-900">{value}</dd>
    </div>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto max-w-5xl">{children}</div>;
}
