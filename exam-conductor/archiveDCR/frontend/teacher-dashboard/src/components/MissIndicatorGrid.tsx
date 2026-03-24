// ---------------------------------------------------------------------------
// MissIndicatorGrid — color-coded student x question matrix.
// ---------------------------------------------------------------------------

import type { MissIndicatorCell, MissState } from '@/api/teacher-api';

const STATE_COLORS: Record<MissState, string> = {
  answered: 'bg-green-400',
  miss_no_strokes: 'bg-red-400',
  miss_sync_failure: 'bg-orange-400',
  miss_pen_inactive: 'bg-yellow-400',
  not_attempted_confirmed: 'bg-gray-400',
};

const STATE_LABELS: Record<MissState, string> = {
  answered: 'Answered',
  miss_no_strokes: 'No strokes',
  miss_sync_failure: 'Sync failure',
  miss_pen_inactive: 'Pen inactive',
  not_attempted_confirmed: 'Not attempted',
};

interface MissIndicatorGridProps {
  students: string[];
  questions: string[];
  cells: MissIndicatorCell[];
}

export function MissIndicatorGrid({
  students,
  questions,
  cells,
}: MissIndicatorGridProps) {
  const cellMap = new Map<string, MissState>();
  for (const c of cells) {
    cellMap.set(`${c.student_id}::${c.question_id}`, c.state);
  }

  return (
    <div className="space-y-4">
      {/* Legend */}
      <div className="flex flex-wrap gap-3 text-xs">
        {(Object.entries(STATE_COLORS) as [MissState, string][]).map(
          ([state, color]) => (
            <div key={state} className="flex items-center gap-1.5">
              <span className={`inline-block h-3 w-3 rounded ${color}`} />
              <span className="text-gray-600">{STATE_LABELS[state]}</span>
            </div>
          ),
        )}
      </div>

      {/* Grid */}
      <div className="overflow-x-auto">
        <table className="text-xs">
          <thead>
            <tr>
              <th className="px-2 py-1 text-left font-medium text-gray-500">
                Student
              </th>
              {questions.map((q) => (
                <th
                  key={q}
                  className="px-2 py-1 text-center font-medium text-gray-500"
                >
                  {q}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {students.map((sid) => (
              <tr key={sid}>
                <td className="whitespace-nowrap px-2 py-1 text-gray-700">
                  {sid}
                </td>
                {questions.map((qid) => {
                  const state = cellMap.get(`${sid}::${qid}`);
                  const color = state
                    ? STATE_COLORS[state]
                    : 'bg-gray-100';
                  return (
                    <td key={qid} className="px-1 py-1">
                      <div
                        className={`mx-auto h-5 w-5 rounded ${color}`}
                        title={state ? STATE_LABELS[state] : 'Unknown'}
                      />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
