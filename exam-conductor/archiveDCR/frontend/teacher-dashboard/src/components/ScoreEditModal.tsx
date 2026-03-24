// ---------------------------------------------------------------------------
// ScoreEditModal — inline modal for overriding a question score.
// Requires a mandatory reason field before submission.
// ---------------------------------------------------------------------------

import { useState, type FormEvent } from 'react';

interface ScoreEditModalProps {
  questionId: string;
  currentScore: number;
  maxScore?: number;
  onSubmit: (questionId: string, newScore: number, reason: string) => void;
  onClose: () => void;
  isSubmitting?: boolean;
}

export function ScoreEditModal({
  questionId,
  currentScore,
  maxScore,
  onSubmit,
  onClose,
  isSubmitting = false,
}: ScoreEditModalProps) {
  const [newScore, setNewScore] = useState(currentScore);
  const [reason, setReason] = useState('');

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!reason.trim()) return;
    onSubmit(questionId, newScore, reason.trim());
  }

  const isValid = reason.trim().length > 0 && newScore !== currentScore;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="w-full max-w-md rounded-lg bg-white p-6 shadow-xl">
        <h3 className="text-lg font-semibold text-gray-900">
          Override Score — {questionId}
        </h3>
        <p className="mt-1 text-sm text-gray-500">
          Current score: {currentScore}
        </p>

        <form onSubmit={handleSubmit} className="mt-4 space-y-4">
          <div>
            <label
              htmlFor="new-score"
              className="block text-sm font-medium text-gray-700"
            >
              New Score
            </label>
            <input
              id="new-score"
              type="number"
              step="0.5"
              min={0}
              max={maxScore}
              value={newScore}
              onChange={(e) => setNewScore(Number(e.target.value))}
              className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm
                         focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
          </div>

          <div>
            <label
              htmlFor="reason"
              className="block text-sm font-medium text-gray-700"
            >
              Reason <span className="text-red-500">*</span>
            </label>
            <textarea
              id="reason"
              rows={3}
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Explain why this score is being changed..."
              className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm
                         focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
          </div>

          <div className="flex justify-end gap-3">
            <button
              type="button"
              onClick={onClose}
              className="rounded-md px-4 py-2 text-sm text-gray-600 hover:bg-gray-100"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!isValid || isSubmitting}
              className="rounded-md bg-brand-600 px-4 py-2 text-sm font-medium text-white
                         hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isSubmitting ? 'Saving...' : 'Save Override'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
