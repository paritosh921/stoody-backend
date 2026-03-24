// ---------------------------------------------------------------------------
// StatusBadge — color-coded pill for exam / score / objection statuses.
// ---------------------------------------------------------------------------

import type { ReactNode } from 'react';

type Variant = 'success' | 'warning' | 'error' | 'info' | 'neutral';

const VARIANT_CLASSES: Record<Variant, string> = {
  success: 'bg-green-100 text-green-800',
  warning: 'bg-yellow-100 text-yellow-800',
  error: 'bg-red-100 text-red-800',
  info: 'bg-blue-100 text-blue-800',
  neutral: 'bg-gray-100 text-gray-700',
};

const STATUS_VARIANT: Record<string, Variant> = {
  // Exam states
  draft: 'neutral',
  scheduled: 'info',
  in_progress: 'warning',
  completed: 'success',
  cancelled: 'error',
  // Score states
  ai_draft: 'info',
  teacher_reviewed: 'warning',
  finalized: 'success',
  locked: 'neutral',
  // Objection states
  filed: 'warning',
  assigned: 'info',
  reviewing: 'info',
  resolved: 'success',
  rejected: 'error',
  // Plagiarism severity
  low: 'neutral',
  medium: 'warning',
  high: 'error',
};

interface StatusBadgeProps {
  status: string;
  variant?: Variant;
  children?: ReactNode;
}

export function StatusBadge({ status, variant, children }: StatusBadgeProps) {
  const resolved = variant ?? STATUS_VARIANT[status] ?? 'neutral';
  const label = children ?? status.replace(/_/g, ' ');

  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ${VARIANT_CLASSES[resolved]}`}
    >
      {label}
    </span>
  );
}
