// ---------------------------------------------------------------------------
// RubricQuestionRow — editable row for one question in the rubric editor.
// ---------------------------------------------------------------------------

export interface RubricStep {
  name: string;
  marks: number;
}

export interface RubricQuestion {
  question_number: number;
  max_marks: number;
  answer_type: string;
  steps: RubricStep[];
}

interface Props {
  question: RubricQuestion;
  answerTypes: readonly string[];
  onChange: (q: RubricQuestion) => void;
  onRemove: () => void;
}

export function RubricQuestionRow({ question, answerTypes, onChange, onRemove }: Props) {
  const q = question;
  const stepSum = q.steps.reduce((s, st) => s + st.marks, 0);
  const sumMismatch = q.steps.length > 0 && stepSum !== q.max_marks;

  function updateField<K extends keyof RubricQuestion>(field: K, value: RubricQuestion[K]) {
    onChange({ ...q, [field]: value });
  }

  function updateStep(idx: number, step: RubricStep) {
    const next = q.steps.map((s, i) => (i === idx ? step : s));
    onChange({ ...q, steps: next });
  }

  function addStep() {
    onChange({ ...q, steps: [...q.steps, { name: '', marks: 0 }] });
  }

  function removeStep(idx: number) {
    onChange({ ...q, steps: q.steps.filter((_, i) => i !== idx) });
  }

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex items-center gap-3">
        <span className="text-sm font-semibold text-gray-700 w-8">
          Q{q.question_number}
        </span>

        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-500">Max Marks</label>
          <input type="number" min={1} value={q.max_marks}
            onChange={(e) => updateField('max_marks', +e.target.value)}
            className="w-16 rounded-md border border-gray-300 px-2 py-1 text-sm" />
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-500">Answer Type</label>
          <select value={q.answer_type}
            onChange={(e) => updateField('answer_type', e.target.value)}
            className="rounded-md border border-gray-300 px-2 py-1 text-sm">
            {answerTypes.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        <div className="ml-auto flex gap-2">
          <button onClick={addStep} type="button"
            className="text-xs text-brand-600 hover:underline">
            + Step
          </button>
          <button onClick={onRemove} type="button"
            className="text-xs text-red-500 hover:underline">
            Remove
          </button>
        </div>
      </div>

      {q.steps.length > 0 && (
        <div className="mt-3 ml-8 space-y-2">
          {q.steps.map((step, idx) => (
            <div key={idx} className="flex items-center gap-2">
              <input type="text" value={step.name} placeholder="Step name"
                onChange={(e) => updateStep(idx, { ...step, name: e.target.value })}
                className="flex-1 rounded-md border border-gray-300 px-2 py-1 text-sm" />
              <input type="number" min={0} value={step.marks}
                onChange={(e) => updateStep(idx, { ...step, marks: +e.target.value })}
                className="w-16 rounded-md border border-gray-300 px-2 py-1 text-sm" />
              <button onClick={() => removeStep(idx)} type="button"
                className="text-xs text-red-400 hover:text-red-600">
                x
              </button>
            </div>
          ))}
          <p className={`text-xs ${sumMismatch ? 'text-red-500' : 'text-gray-400'}`}>
            Step total: {stepSum} / {q.max_marks}
            {sumMismatch && ' (mismatch)'}
          </p>
        </div>
      )}
    </div>
  );
}
