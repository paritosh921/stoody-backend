// ---------------------------------------------------------------------------
// CreateExamPage — form to create a new exam with full configuration.
// ---------------------------------------------------------------------------

import { useState, type FormEvent, type ChangeEvent } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { createExam } from '@/api/teacher-api';

const VARIANT_OPTIONS = ['A', 'B', 'C', 'D'] as const;

interface ExamFormData {
  title: string;
  subject_id: string;
  class_id: string;
  section_id: string;
  scheduled_at: string;
  duration_min: number;
  question_count: number;
  total_marks: number;
  negative_marking: boolean;
  variants: string[];
}

export function CreateExamPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState<ExamFormData>({
    title: '',
    subject_id: '',
    class_id: '',
    section_id: '',
    scheduled_at: '',
    duration_min: 60,
    question_count: 10,
    total_marks: 100,
    negative_marking: false,
    variants: ['A'],
  });
  const [qpFile, setQpFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function update<K extends keyof ExamFormData>(field: K, value: ExamFormData[K]) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  function toggleVariant(v: string) {
    setForm((prev) => {
      const has = prev.variants.includes(v);
      const next = has
        ? prev.variants.filter((x) => x !== v)
        : [...prev.variants, v];
      return { ...prev, variants: next.length ? next : prev.variants };
    });
  }

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0] ?? null;
    if (file && file.type !== 'application/pdf') {
      setError('Only PDF files are accepted for question papers.');
      return;
    }
    setQpFile(file);
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!form.title || !form.subject_id || !form.section_id || !form.scheduled_at) return;
    setSubmitting(true);
    setError(null);
    try {
      // Payload matches svc-exam-orch CreateExamBody exactly
      await createExam(form as unknown as Record<string, unknown>);
      // NOTE: If qpFile is set, the upload URL is returned after creation.
      // A follow-up upload step would use the returned exam_id + presigned URL.
      navigate('/exams');
    } catch {
      setError('Failed to create exam. Please try again.');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl">
      <Link to="/exams" className="text-sm text-brand-600 hover:underline">
        &larr; Back to Exams
      </Link>

      <h1 className="mt-4 text-xl font-semibold text-gray-900">
        Create New Exam
      </h1>

      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}

      <form onSubmit={handleSubmit} className="mt-6 space-y-5">
        <Field label="Title" required>
          <input type="text" value={form.title}
            onChange={(e) => update('title', e.target.value)}
            className="input-field" placeholder="Midterm Mathematics" />
        </Field>

        <div className="grid grid-cols-2 gap-4">
          <Field label="Subject ID" required>
            <input type="text" value={form.subject_id}
              onChange={(e) => update('subject_id', e.target.value)}
              className="input-field" placeholder="math-101" />
          </Field>
          <Field label="Class ID" required>
            <input type="text" value={form.class_id}
              onChange={(e) => update('class_id', e.target.value)}
              className="input-field" placeholder="class-10" />
          </Field>
          <Field label="Section ID" required>
            <input type="text" value={form.section_id}
              onChange={(e) => update('section_id', e.target.value)}
              className="input-field" placeholder="section-a" />
          </Field>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Field label="Scheduled At" required>
            <input type="datetime-local" value={form.scheduled_at}
              onChange={(e) => update('scheduled_at', e.target.value)}
              className="input-field" />
          </Field>
          <Field label="Duration (minutes)">
            <input type="number" min={10} max={300} value={form.duration_min}
              onChange={(e) => update('duration_min', +e.target.value)}
              className="input-field" />
          </Field>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Field label="Question Count">
            <input type="number" min={1} max={200} value={form.question_count}
              onChange={(e) => update('question_count', +e.target.value)}
              className="input-field" />
          </Field>
          <Field label="Total Marks">
            <input type="number" min={1} value={form.total_marks}
              onChange={(e) => update('total_marks', +e.target.value)}
              className="input-field" />
          </Field>
        </div>

        <NegativeMarkingSection form={form} update={update} />
        <VariantSection variants={form.variants} toggle={toggleVariant} />

        <Field label="Question Paper (PDF)">
          <input type="file" accept="application/pdf" onChange={handleFileChange}
            className="block w-full text-sm text-gray-500 file:mr-4 file:rounded-md
              file:border-0 file:bg-brand-50 file:px-4 file:py-2 file:text-sm
              file:font-medium file:text-brand-700 hover:file:bg-brand-100" />
          {qpFile && (
            <p className="mt-1 text-xs text-gray-500">{qpFile.name}</p>
          )}
        </Field>

        <button type="submit" disabled={submitting}
          className="rounded-md bg-brand-600 px-5 py-2 text-sm font-medium text-white
                     hover:bg-brand-700 disabled:opacity-50">
          {submitting ? 'Creating...' : 'Create Exam'}
        </button>
      </form>
    </div>
  );
}

/* --- Sub-components ---------------------------------------------------- */

function NegativeMarkingSection({
  form,
  update,
}: {
  form: { negative_marking: boolean };
  update: (field: 'negative_marking', v: boolean) => void;
}) {
  return (
    <div className="flex items-center gap-4">
      <label className="flex items-center gap-2 text-sm text-gray-700">
        <input type="checkbox" checked={form.negative_marking}
          onChange={(e) => update('negative_marking', e.target.checked)}
          className="h-4 w-4 rounded border-gray-300 text-brand-600
                     focus:ring-brand-500" />
        Enable Negative Marking
      </label>
      {form.negative_marking && (
        <span className="text-xs text-gray-500">
          (Deduction value configured in rubric per question)
        </span>
      )}
    </div>
  );
}

function VariantSection({
  variants,
  toggle,
}: {
  variants: string[];
  toggle: (v: string) => void;
}) {
  return (
    <Field label="Variants">
      <div className="flex gap-4">
        {VARIANT_OPTIONS.map((v) => (
          <label key={v} className="flex items-center gap-1.5 text-sm text-gray-700">
            <input type="checkbox" checked={variants.includes(v)}
              onChange={() => toggle(v)}
              className="h-4 w-4 rounded border-gray-300 text-brand-600
                         focus:ring-brand-500" />
            {v}
          </label>
        ))}
      </div>
    </Field>
  );
}

function Field({
  label,
  required,
  children,
}: {
  label: string;
  required?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700">
        {label} {required && <span className="text-red-500">*</span>}
      </label>
      <div className="mt-1 [&_.input-field]:w-full [&_.input-field]:rounded-md [&_.input-field]:border [&_.input-field]:border-gray-300 [&_.input-field]:px-3 [&_.input-field]:py-2 [&_.input-field]:text-sm [&_.input-field]:focus:border-brand-500 [&_.input-field]:focus:outline-none [&_.input-field]:focus:ring-1 [&_.input-field]:focus:ring-brand-500">
        {children}
      </div>
    </div>
  );
}
