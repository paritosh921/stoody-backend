// ---------------------------------------------------------------------------
// LeaderboardPage — ranked student leaderboard for an exam.
// ---------------------------------------------------------------------------

import { useQuery } from '@tanstack/react-query';
import { useParams, Link } from 'react-router-dom';
import { getLeaderboard } from '@/api/teacher-api';

export function LeaderboardPage() {
  const { examId } = useParams<{ examId: string }>();

  const { data: entries, isLoading, error } = useQuery({
    queryKey: ['leaderboard', examId],
    queryFn: () => getLeaderboard(examId!),
    enabled: !!examId,
    select: (res) => res.data?.items ?? [],
  });

  if (isLoading) return <Shell>Loading leaderboard...</Shell>;
  if (error) return <Shell>Failed to load leaderboard.</Shell>;

  return (
    <Shell>
      <Link to="/analytics" className="text-sm text-brand-600 hover:underline">
        &larr; Analytics
      </Link>

      <h1 className="mt-4 text-xl font-semibold text-gray-900">
        Leaderboard
      </h1>

      <div className="mt-4 overflow-hidden rounded-lg border border-gray-200">
        <table className="min-w-full divide-y divide-gray-200 text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-2 text-left font-medium text-gray-600">
                Rank
              </th>
              <th className="px-4 py-2 text-left font-medium text-gray-600">
                Student
              </th>
              <th className="px-4 py-2 text-right font-medium text-gray-600">
                Score
              </th>
              <th className="px-4 py-2 text-right font-medium text-gray-600">
                Percentile
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 bg-white">
            {entries?.map((e) => (
              <tr key={e.student_id}>
                <td className="px-4 py-2 font-medium text-gray-900">
                  #{e.rank}
                </td>
                <td className="px-4 py-2 text-gray-800">{e.student_name}</td>
                <td className="px-4 py-2 text-right tabular-nums">
                  {e.total_score}
                </td>
                <td className="px-4 py-2 text-right tabular-nums text-gray-500">
                  {e.percentile}%
                </td>
              </tr>
            ))}
            {entries?.length === 0 && (
              <tr>
                <td colSpan={4} className="px-4 py-8 text-center text-gray-400">
                  No data available.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto max-w-3xl">{children}</div>;
}
