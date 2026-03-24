/// HTTP client for the Student BFF endpoints.
///
/// All calls are routed through [NetworkService] which handles auth headers,
/// base URL resolution, retries, and error mapping.
///
/// Endpoints implemented (student-bff routes):
///   GET  /api/v1/student/exams
///   GET  /api/v1/student/exams/{exam_id}/score
///   GET  /api/v1/student/exams/{exam_id}/questions
///   GET  /api/v1/student/exams/{exam_id}/questions/{qid}/answer
///   GET  /api/v1/student/objections
///   POST /api/v1/student/exams/{exam_id}/objections
///   GET  /api/v1/student/exams/{exam_id}/chat/{teacher_id}
///   POST /api/v1/student/exams/{exam_id}/chat/{teacher_id}
///   GET  /api/v1/student/performance/history
///   GET  /api/v1/student/performance/trends
///   GET  /api/v1/student/performance/strengths
library;

import 'package:exampen_mobile/core/network_service.dart';

// ---------------------------------------------------------------------------
// Request / response models
// ---------------------------------------------------------------------------

enum ExamStatus {
  upcoming,
  scoresPending,
  published,
  objectionWindowOpen,
  locked;

  static ExamStatus fromString(String value) {
    switch (value) {
      case 'scores_pending':
        return ExamStatus.scoresPending;
      case 'objection_window_open':
        return ExamStatus.objectionWindowOpen;
      default:
        return ExamStatus.values.firstWhere(
          (e) => e.name == value,
          orElse: () => ExamStatus.upcoming,
        );
    }
  }

  String get displayLabel {
    switch (this) {
      case ExamStatus.upcoming:
        return 'Upcoming';
      case ExamStatus.scoresPending:
        return 'Scores Pending';
      case ExamStatus.published:
        return 'Published';
      case ExamStatus.objectionWindowOpen:
        return 'Objection Window Open';
      case ExamStatus.locked:
        return 'Locked';
    }
  }
}

class StudentExamCard {
  final String examId;
  final String title;
  final String? subjectName;
  final DateTime scheduledAt;
  final ExamStatus status;

  const StudentExamCard({
    required this.examId,
    required this.title,
    this.subjectName,
    required this.scheduledAt,
    required this.status,
  });

  factory StudentExamCard.fromJson(Map<String, dynamic> json) {
    return StudentExamCard(
      examId: json['exam_id'] as String? ?? '',
      title: json['title'] as String? ?? '',
      subjectName: json['subject_name'] as String?,
      scheduledAt: DateTime.tryParse(json['scheduled_at'] as String? ?? '') ??
          DateTime.now(),
      status: ExamStatus.fromString(json['status'] as String? ?? ''),
    );
  }
}

class StudentQuestionScore {
  final String questionId;
  final double marksObtained;
  final double maxMarks;
  final double? aiConfidence;
  final String? missIndicator;

  const StudentQuestionScore({
    required this.questionId,
    required this.marksObtained,
    required this.maxMarks,
    this.aiConfidence,
    this.missIndicator,
  });

  factory StudentQuestionScore.fromJson(Map<String, dynamic> json) {
    return StudentQuestionScore(
      questionId: json['question_id'] as String? ?? '',
      marksObtained: (json['marks_obtained'] as num?)?.toDouble() ?? 0,
      maxMarks: (json['max_marks'] as num?)?.toDouble() ?? 0,
      aiConfidence: (json['ai_confidence'] as num?)?.toDouble(),
      missIndicator: json['miss_indicator'] as String?,
    );
  }
}

class StudentScoreView {
  final String examId;
  final double totalScore;
  final double percentage;
  final double percentile;
  final String? passFail;
  final List<StudentQuestionScore> questions;

  const StudentScoreView({
    required this.examId,
    required this.totalScore,
    required this.percentage,
    required this.percentile,
    this.passFail,
    this.questions = const [],
  });

  factory StudentScoreView.fromJson(Map<String, dynamic> json) {
    return StudentScoreView(
      examId: json['exam_id'] as String? ?? '',
      totalScore: (json['total_score'] as num?)?.toDouble() ?? 0,
      percentage: (json['percentage'] as num?)?.toDouble() ?? 0,
      percentile: (json['percentile'] as num?)?.toDouble() ?? 0,
      passFail: json['pass_fail'] as String?,
      questions: (json['questions'] as List<dynamic>?)
              ?.map((e) =>
                  StudentQuestionScore.fromJson(e as Map<String, dynamic>))
              .toList(growable: false) ??
          const [],
    );
  }

  bool get passed => passFail == 'pass';
}

class AnswerInsight {
  final String questionId;
  final String answerImageUri;
  final String recognizedText;
  final double confidence;
  final List<String> stepBreakdown;
  final String? feedback;

  const AnswerInsight({
    required this.questionId,
    required this.answerImageUri,
    required this.recognizedText,
    required this.confidence,
    this.stepBreakdown = const [],
    this.feedback,
  });

  factory AnswerInsight.fromJson(Map<String, dynamic> json) {
    return AnswerInsight(
      questionId: json['question_id'] as String? ?? '',
      answerImageUri: json['answer_image_uri'] as String? ?? '',
      recognizedText: json['recognized_text'] as String? ?? '',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0,
      stepBreakdown: (json['step_breakdown'] as List<dynamic>?)
              ?.map((e) => e as String)
              .toList(growable: false) ??
          const [],
      feedback: json['feedback'] as String?,
    );
  }
}

enum ObjectionStatus {
  filed,
  assigned,
  reviewing,
  resolved,
  escalated;

  static ObjectionStatus fromString(String value) {
    return ObjectionStatus.values.firstWhere(
      (e) => e.name == value,
      orElse: () => ObjectionStatus.filed,
    );
  }

  String get displayLabel {
    switch (this) {
      case ObjectionStatus.filed:
        return 'Filed';
      case ObjectionStatus.assigned:
        return 'Assigned';
      case ObjectionStatus.reviewing:
        return 'Under Review';
      case ObjectionStatus.resolved:
        return 'Resolved';
      case ObjectionStatus.escalated:
        return 'Escalated';
    }
  }
}

class StudentObjection {
  final String objectionId;
  final String examId;
  final String questionId;
  final ObjectionStatus status;
  final String? objectionText;
  final String? resolutionReason;
  final double? newScore;

  const StudentObjection({
    required this.objectionId,
    required this.examId,
    required this.questionId,
    required this.status,
    this.objectionText,
    this.resolutionReason,
    this.newScore,
  });

  factory StudentObjection.fromJson(Map<String, dynamic> json) {
    return StudentObjection(
      objectionId: json['objection_id'] as String? ?? '',
      examId: json['exam_id'] as String? ?? '',
      questionId: json['question_id'] as String? ?? '',
      status: ObjectionStatus.fromString(json['status'] as String? ?? ''),
      objectionText: json['objection_text'] as String?,
      resolutionReason: json['resolution_reason'] as String?,
      newScore: (json['new_score'] as num?)?.toDouble(),
    );
  }
}

class CreateObjectionRequest {
  final String examId;
  final String questionId;
  final String objectionText;

  const CreateObjectionRequest({
    required this.examId,
    required this.questionId,
    required this.objectionText,
  });

  Map<String, dynamic> toJson() => {
        'exam_id': examId,
        'question_id': questionId,
        'objection_text': objectionText,
      };
}

class ChatMessage {
  final String messageId;
  final String senderId;
  final String content;
  final String? attachmentUri;
  final DateTime sentAt;
  final DateTime? readAt;

  const ChatMessage({
    required this.messageId,
    required this.senderId,
    required this.content,
    this.attachmentUri,
    required this.sentAt,
    this.readAt,
  });

  factory ChatMessage.fromJson(Map<String, dynamic> json) {
    return ChatMessage(
      messageId: json['message_id'] as String? ?? '',
      senderId: json['sender_id'] as String? ?? '',
      content: json['content'] as String? ?? '',
      attachmentUri: json['attachment_uri'] as String?,
      sentAt: DateTime.tryParse(json['sent_at'] as String? ?? '') ??
          DateTime.now(),
      readAt: json['read_at'] != null
          ? DateTime.tryParse(json['read_at'] as String)
          : null,
    );
  }
}

class PerformanceHistoryEntry {
  final String examId;
  final double score;
  final double percentile;

  const PerformanceHistoryEntry({
    required this.examId,
    required this.score,
    required this.percentile,
  });

  factory PerformanceHistoryEntry.fromJson(Map<String, dynamic> json) {
    return PerformanceHistoryEntry(
      examId: json['exam_id'] as String? ?? '',
      score: (json['score'] as num?)?.toDouble() ?? 0,
      percentile: (json['percentile'] as num?)?.toDouble() ?? 0,
    );
  }
}

class PerformanceView {
  final List<PerformanceHistoryEntry> history;
  final List<String> strengths;
  final List<String> weaknesses;

  const PerformanceView({
    this.history = const [],
    this.strengths = const [],
    this.weaknesses = const [],
  });

  factory PerformanceView.fromJson(Map<String, dynamic> json) {
    return PerformanceView(
      history: (json['history'] as List<dynamic>?)
              ?.map((e) =>
                  PerformanceHistoryEntry.fromJson(e as Map<String, dynamic>))
              .toList(growable: false) ??
          const [],
      strengths: (json['strengths'] as List<dynamic>?)
              ?.map((e) => e as String)
              .toList(growable: false) ??
          const [],
      weaknesses: (json['weaknesses'] as List<dynamic>?)
              ?.map((e) => e as String)
              .toList(growable: false) ??
          const [],
    );
  }
}

// ---------------------------------------------------------------------------
// API client
// ---------------------------------------------------------------------------

class StudentApi {
  final NetworkService _network;

  StudentApi(this._network);

  // -- Exams ----------------------------------------------------------------

  Future<List<StudentExamCard>> listExams() async {
    final json = await _network.get<Map<String, dynamic>>(
      '/api/v1/student/exams',
    );
    final items = json['items'] as List<dynamic>? ?? [];
    return items
        .map((e) => StudentExamCard.fromJson(e as Map<String, dynamic>))
        .toList(growable: false);
  }

  // -- Scores ---------------------------------------------------------------

  Future<StudentScoreView> getScores(String examId) async {
    final json = await _network.get<Map<String, dynamic>>(
      '/api/v1/student/exams/$examId/score',
    );
    return StudentScoreView.fromJson(json);
  }

  // -- Answer insight -------------------------------------------------------

  Future<AnswerInsight> getAnswerInsight(
    String examId,
    String questionId,
  ) async {
    final json = await _network.get<Map<String, dynamic>>(
      '/api/v1/student/exams/$examId/questions/$questionId/answer',
    );
    return AnswerInsight.fromJson(json);
  }

  // -- Objections -----------------------------------------------------------

  Future<List<StudentObjection>> listObjections() async {
    final json = await _network.get<Map<String, dynamic>>(
      '/api/v1/student/objections',
    );
    final items = json['items'] as List<dynamic>? ?? [];
    return items
        .map((e) => StudentObjection.fromJson(e as Map<String, dynamic>))
        .toList(growable: false);
  }

  Future<StudentObjection> fileObjection(CreateObjectionRequest body) async {
    final json = await _network.post<Map<String, dynamic>>(
      '/api/v1/student/exams/${body.examId}/objections',
      body: body.toJson(),
    );
    return StudentObjection.fromJson(json);
  }

  // -- Chat -----------------------------------------------------------------

  Future<List<ChatMessage>> getChatThread(
    String examId,
    String teacherId,
  ) async {
    final json = await _network.get<Map<String, dynamic>>(
      '/api/v1/student/exams/$examId/chat/$teacherId',
    );
    final items = json['items'] as List<dynamic>? ?? [];
    return items
        .map((e) => ChatMessage.fromJson(e as Map<String, dynamic>))
        .toList(growable: false);
  }

  Future<ChatMessage> sendMessage(
    String examId,
    String teacherId, {
    required String content,
    String? attachmentUri,
  }) async {
    final body = <String, dynamic>{'content': content};
    if (attachmentUri != null) body['attachment_uri'] = attachmentUri;

    final json = await _network.post<Map<String, dynamic>>(
      '/api/v1/student/exams/$examId/chat/$teacherId',
      body: body,
    );
    return ChatMessage.fromJson(json);
  }

  // -- Performance ----------------------------------------------------------

  /// Fetch combined history + strengths/weaknesses.
  Future<PerformanceView> getPerformanceHistory() async {
    final json = await _network.get<Map<String, dynamic>>(
      '/api/v1/student/performance/history',
    );
    return PerformanceView.fromJson(json);
  }

  /// Fetch trend data for charts.
  Future<Map<String, dynamic>> getPerformanceTrends() async {
    return await _network.get<Map<String, dynamic>>(
      '/api/v1/student/performance/trends',
    );
  }

  /// Fetch AI-generated strengths/weaknesses.
  Future<Map<String, dynamic>> getPerformanceStrengths() async {
    return await _network.get<Map<String, dynamic>>(
      '/api/v1/student/performance/strengths',
    );
  }

  /// Question-wise breakdown for an exam.
  Future<Map<String, dynamic>> getQuestionBreakdown(String examId) async {
    return await _network.get<Map<String, dynamic>>(
      '/api/v1/student/exams/$examId/questions',
    );
  }
}
