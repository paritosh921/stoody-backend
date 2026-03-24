/** Analytics types: leaderboard, class stats, performance trends, exports. */

import type { ExportFormat } from "./enums";

/** Single row in the exam leaderboard. */
export interface LeaderboardRow {
  rank: number;
  student_id: string;
  student_name?: string;
  score: number;
  percentile: number;
}

/** Per-question difficulty metric within class stats. */
export interface QuestionDifficulty {
  question_id: string;
  avg_score: number;
}

/** Class-level statistical summary for an exam. */
export interface ClassStats {
  mean: number;
  median: number;
  std_dev: number;
  pass_rate: number;
  question_difficulty?: QuestionDifficulty[];
}

/** Single exam entry in a student's performance history. */
export interface ExamPerformanceEntry {
  exam_id: string;
  score: number;
  percentile: number;
}

/** Longitudinal performance data for a student. */
export interface StudentPerformance {
  student_id: string;
  history: ExamPerformanceEntry[];
  strengths?: string[];
  weaknesses?: string[];
}

/** Metadata for a generated analytics export file. */
export interface ExportResult {
  exam_id: string;
  format: ExportFormat;
  download_uri: string;
}

/** Student-facing historical performance and trend data. */
export interface PerformanceView {
  history: ExamPerformanceEntry[];
  strengths: string[];
  weaknesses: string[];
}
