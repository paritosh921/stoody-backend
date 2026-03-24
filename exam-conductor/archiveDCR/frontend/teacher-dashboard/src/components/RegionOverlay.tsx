// ---------------------------------------------------------------------------
// RegionOverlay — renders a draggable bounding box on the answer sheet image.
// Uses the canonical QuestionRegion type from teacher-types.ts.
// ---------------------------------------------------------------------------

import type { QuestionRegion } from '@/api/teacher-types';
export type { QuestionRegion };

interface Props {
  region: QuestionRegion;
  onRemove?: () => void;
}

export function RegionOverlay({ region, onRemove }: Props) {
  return (
    <div
      className="absolute border-2 border-brand-500 bg-brand-500/10"
      style={{
        left: `${region.x_pct}%`,
        top: `${region.y_pct}%`,
        width: `${region.width_pct}%`,
        height: `${region.height_pct}%`,
      }}
    >
      <span className="absolute -top-5 left-0 rounded bg-brand-600 px-1.5 py-0.5
                        text-[10px] font-bold text-white leading-none">
        Q{region.question_number}
      </span>
      {onRemove && (
        <button
          onClick={(e) => { e.stopPropagation(); onRemove(); }}
          className="absolute -top-2 -right-2 flex h-4 w-4 items-center justify-center
                     rounded-full bg-red-500 text-[10px] text-white leading-none
                     hover:bg-red-600"
        >
          &times;
        </button>
      )}
    </div>
  );
}
