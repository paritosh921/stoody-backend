// ---------------------------------------------------------------------------
// ScoreTable — sortable, filterable data table for class scores.
// ---------------------------------------------------------------------------

import { useState, useMemo, type ReactNode } from 'react';

export interface Column<T> {
  key: string;
  header: string;
  render: (row: T) => ReactNode;
  sortValue?: (row: T) => number | string;
}

interface ScoreTableProps<T> {
  columns: Column<T>[];
  data: T[];
  rowKey: (row: T) => string;
  searchPlaceholder?: string;
  searchAccessor?: (row: T) => string;
  onRowClick?: (row: T) => void;
}

type SortDir = 'asc' | 'desc';

export function ScoreTable<T>({
  columns,
  data,
  rowKey,
  searchPlaceholder = 'Search...',
  searchAccessor,
  onRowClick,
}: ScoreTableProps<T>) {
  const [search, setSearch] = useState('');
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  const filtered = useMemo(() => {
    if (!search || !searchAccessor) return data;
    const q = search.toLowerCase();
    return data.filter((r) => searchAccessor(r).toLowerCase().includes(q));
  }, [data, search, searchAccessor]);

  const sorted = useMemo(() => {
    if (!sortCol) return filtered;
    const col = columns.find((c) => c.key === sortCol);
    if (!col?.sortValue) return filtered;
    const dir = sortDir === 'asc' ? 1 : -1;
    return [...filtered].sort((a, b) => {
      const va = col.sortValue!(a);
      const vb = col.sortValue!(b);
      if (typeof va === 'number' && typeof vb === 'number')
        return (va - vb) * dir;
      return String(va).localeCompare(String(vb)) * dir;
    });
  }, [filtered, sortCol, sortDir, columns]);

  function handleSort(key: string) {
    if (sortCol === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortCol(key);
      setSortDir('asc');
    }
  }

  return (
    <div className="space-y-3">
      {searchAccessor && (
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={searchPlaceholder}
          className="w-full max-w-xs rounded-md border border-gray-300 px-3 py-1.5 text-sm
                     focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
        />
      )}

      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="min-w-full divide-y divide-gray-200 text-sm">
          <thead className="bg-gray-50">
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  onClick={() => col.sortValue && handleSort(col.key)}
                  className={`px-4 py-2 text-left font-medium text-gray-600 ${
                    col.sortValue ? 'cursor-pointer select-none hover:text-gray-900' : ''
                  }`}
                >
                  {col.header}
                  {sortCol === col.key && (sortDir === 'asc' ? ' ^' : ' v')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 bg-white">
            {sorted.map((row) => (
              <tr
                key={rowKey(row)}
                onClick={() => onRowClick?.(row)}
                className={onRowClick ? 'cursor-pointer hover:bg-gray-50' : ''}
              >
                {columns.map((col) => (
                  <td key={col.key} className="whitespace-nowrap px-4 py-2">
                    {col.render(row)}
                  </td>
                ))}
              </tr>
            ))}
            {sorted.length === 0 && (
              <tr>
                <td
                  colSpan={columns.length}
                  className="px-4 py-8 text-center text-gray-400"
                >
                  No data found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
