// ---------------------------------------------------------------------------
// QuestionRegionEditorPage — draw bounding boxes on answer sheet layout image.
// ---------------------------------------------------------------------------

import { useState, useRef, useCallback, type MouseEvent } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { getExamDetail, getQuestionRegions, saveQuestionRegions } from '@/api/teacher-api';
import { RegionOverlay, type QuestionRegion } from '@/components/RegionOverlay';

export function QuestionRegionEditorPage() {
  const { examId } = useParams<{ examId: string }>();
  const containerRef = useRef<HTMLDivElement>(null);

  const { data: exam } = useQuery({
    queryKey: ['exam', examId],
    queryFn: () => getExamDetail(examId!),
    enabled: !!examId,
    select: (res) => res.data,
  });

  const { data: savedRegions } = useQuery({
    queryKey: ['question-regions', examId],
    queryFn: () => getQuestionRegions(examId!),
    enabled: !!examId,
    select: (res) => res.data?.regions ?? [],
  });

  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [regions, setRegions] = useState<QuestionRegion[]>([]);
  const [drawing, setDrawing] = useState<{ startX: number; startY: number } | null>(null);
  const [currentRect, setCurrentRect] = useState<QuestionRegion | null>(null);
  const [nextQuestion, setNextQuestion] = useState(1);
  const [initialized, setInitialized] = useState(false);

  if (savedRegions && !initialized) {
    setRegions(savedRegions);
    setNextQuestion(savedRegions.length + 1);
    setInitialized(true);
  }

  const save = useMutation({
    mutationFn: () => saveQuestionRegions(examId!, regions),
  });

  function handleImageUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImageUrl(url);
  }

  const getRelativePos = useCallback((e: MouseEvent) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return {
      x: ((e.clientX - rect.left) / rect.width) * 100,
      y: ((e.clientY - rect.top) / rect.height) * 100,
    };
  }, []);

  function handleMouseDown(e: MouseEvent) {
    if (!imageUrl) return;
    const pos = getRelativePos(e);
    setDrawing({ startX: pos.x, startY: pos.y });
  }

  function handleMouseMove(e: MouseEvent) {
    if (!drawing) return;
    const pos = getRelativePos(e);
    setCurrentRect({
      question_number: nextQuestion,
      x_pct: Math.min(drawing.startX, pos.x),
      y_pct: Math.min(drawing.startY, pos.y),
      width_pct: Math.abs(pos.x - drawing.startX),
      height_pct: Math.abs(pos.y - drawing.startY),
    });
  }

  function handleMouseUp() {
    if (currentRect && currentRect.width_pct > 1 && currentRect.height_pct > 1) {
      setRegions((prev) => [...prev, currentRect]);
      setNextQuestion((n) => n + 1);
    }
    setDrawing(null);
    setCurrentRect(null);
  }

  function removeRegion(idx: number) {
    setRegions((prev) => prev.filter((_, i) => i !== idx));
  }

  return (
    <div className="mx-auto max-w-5xl">
      <Link to={`/exams/${examId}`} className="text-sm text-brand-600 hover:underline">
        &larr; Back to Exam
      </Link>

      <h1 className="mt-4 text-xl font-semibold text-gray-900">
        Question Region Editor
      </h1>
      <p className="mt-1 text-sm text-gray-500">{exam?.title ?? `Exam ${examId}`}</p>

      {!imageUrl && (
        <div className="mt-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Upload answer sheet layout image
          </label>
          <input type="file" accept="image/*" onChange={handleImageUpload}
            className="block text-sm text-gray-500 file:mr-4 file:rounded-md
              file:border-0 file:bg-brand-50 file:px-4 file:py-2 file:text-sm
              file:font-medium file:text-brand-700 hover:file:bg-brand-100" />
        </div>
      )}

      {imageUrl && (
        <>
          <p className="mt-4 text-xs text-gray-400">
            Click and drag on the image to define question regions.
            Next region: Q{nextQuestion}
          </p>

          <div ref={containerRef} className="relative mt-2 select-none border border-gray-300"
            onMouseDown={handleMouseDown} onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}>
            <img src={imageUrl} alt="Answer sheet layout"
              className="w-full pointer-events-none" draggable={false} />
            {regions.map((r, i) => (
              <RegionOverlay key={i} region={r} onRemove={() => removeRegion(i)} />
            ))}
            {currentRect && <RegionOverlay region={currentRect} />}
          </div>
        </>
      )}

      {/* Region list */}
      {regions.length > 0 && (
        <div className="mt-4">
          <h2 className="text-sm font-medium text-gray-700 mb-2">
            Defined Regions ({regions.length})
          </h2>
          <div className="flex flex-wrap gap-2">
            {regions.map((r, i) => (
              <span key={i} className="inline-flex items-center gap-1 rounded-full
                bg-brand-50 px-3 py-1 text-xs font-medium text-brand-700">
                Q{r.question_number}
                <button onClick={() => removeRegion(i)}
                  className="ml-1 text-brand-400 hover:text-red-500">&times;</button>
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="mt-6 flex gap-3">
        <button onClick={() => save.mutate()}
          disabled={regions.length === 0 || save.isPending}
          className="rounded-md bg-brand-600 px-5 py-2 text-sm font-medium text-white
                     hover:bg-brand-700 disabled:opacity-50">
          {save.isPending ? 'Saving...' : 'Save Regions'}
        </button>
        {save.isSuccess && (
          <span className="self-center text-sm text-green-600">Saved!</span>
        )}
        {save.isError && (
          <span className="self-center text-sm text-red-600">Save failed.</span>
        )}
      </div>
    </div>
  );
}
