/// HTTP client for the Teacher BFF endpoints.
///
/// All calls are routed through [NetworkService] which handles auth headers,
/// base URL resolution, retries, and error mapping.
///
/// Endpoints implemented (teacher-bff routes):
///   GET   /api/v1/teacher/exams
///   GET   /api/v1/teacher/exams/{exam_id}/scores
///   GET   /api/v1/teacher/exams/{exam_id}/students/{student_id}
///   PATCH /api/v1/teacher/exams/{exam_id}/students/{sid}/questions/{qid}
///   POST  /api/v1/teacher/exams/{exam_id}/scores/finalize
///   POST  /api/v1/teacher/exams/{exam_id}/scores/publish
///   GET   /api/v1/teacher/exams/{exam_id}/objections
///   GET   /api/v1/teacher/objections/{objection_id}
///   POST  /api/v1/teacher/objections/{objection_id}/resolve
///   POST  /api/v1/teacher/objections/{objection_id}/escalate
library;

import 'package:exampen_mobile/core/network_service.dart';

// ---------------------------------------------------------------------------
// Request / response models
// ---------------------------------------------------------------------------

class TeacherExamCard {
  final String examId;
  final String title;
  final String subjectId;
  final DateTime scheduledAt;
  final String state;
  final String? classLabel;

  const TeacherExamCard({
    required this.examId,
    required this.title,
    required this.subjectId,
    required this.scheduledAt,
    required this.state,
    this.classLabel,
  });

  factory TeacherExamCard.fromJson(Map<String, dynamic> json) {
    return TeacherExamCard(
      examId: json['exam_id'] as String? ?? '',
      title: json['title'] as String? ?? '',
      subjectId: json['subject_id'] as String? ?? '',
      scheduledAt: DateTime.tryParse(json['scheduled_at'] as String? ?? '') ??
          DateTime.now(),
      state: json['state'] as String? ?? '',
      classLabel: json['class_label'] as String?,
    );
  }
}

class ClassScoreRow {
  final String studentId;
  final String studentName;
  final double totalScore;
  final double? percentile;
  final double aiConfidence;
  final int? missIndicatorCount;
  final int? plagiarismFlagCount;

  const ClassScoreRow({
    required this.studentId,
    required this.studentName,
    required this.totalScore,
    this.percentile,
    required this.aiConfidence,
    this.missIndicatorCount,
    this.plagiarismFlagCount,
  });

  factory ClassScoreRow.fromJson(Map<String, dynamic> json) {
    return ClassScoreRow(
      studentId: json['student_id'] as String? ?? '',
      studentName: json['student_name'] as String? ?? '',
      totalScore: (json['total_score'] as num?)?.toDouble() ?? 0,
      percentile: (json['percentile'] as num?)?.toDouble(),
      aiConfidence: (json['ai_confidence'] as num?)?.toDouble() ?? 0,
      missIndicatorCount: json['miss_indicator_count'] as int?,
      plagiarismFlagCount: json['plagiarism_flag_count'] as int?,
    );
  }
}

class QuestionDetail {
  final String questionId;
  final double currentScore;
  final double confidence;
  final String? recognizedText;
  final String? missIndicator;
  final String? copyImageUri;

  const QuestionDetail({
    required this.questionId,
    required this.currentScore,
    required this.confidence,
    this.recognizedText,
    this.missIndicator,
    this.copyImageUri,
  });

  factory QuestionDetail.fromJson(Map<String, dynamic> json) {
    return QuestionDetail(
      questionId: json['question_id'] as String? ?? '',
      currentScore: (json['current_score'] as num?)?.toDouble() ?? 0,
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0,
      recognizedText: json['recognized_text'] as String?,
      missIndicator: json['miss_indicator'] as String?,
      copyImageUri: json['copy_image_uri'] as String?,
    );
  }
}

class TeacherStudentDetail {
  final String studentId;
  final String studentName;
  final double totalScore;
  final List<String> answerPages;
  final List<QuestionDetail> questions;

  const TeacherStudentDetail({
    required this.studentId,
    required this.studentName,
    required this.totalScore,
    this.answerPages = const [],
    this.questions = const [],
  });

  factory TeacherStudentDetail.fromJson(Map<String, dynamic> json) {
    return TeacherStudentDetail(
      studentId: json['student_id'] as String? ?? '',
      studentName: json['student_name'] as String? ?? '',
      totalScore: (json['total_score'] as num?)?.toDouble() ?? 0,
      answerPages: (json['answer_pages'] as List<dynamic>?)
              ?.map((e) => e as String)
              .toList(growable: false) ??
          const [],
      questions: (json['questions'] as List<dynamic>?)
              ?.map((e) => QuestionDetail.fromJson(e as Map<String, dynamic>))
              .toList(growable: false) ??
          const [],
    );
  }
}

class TeacherScoreOverrideRequest {
  final String questionId;
  final double newScore;
  final String reason;

  const TeacherScoreOverrideRequest({
    required this.questionId,
    required this.newScore,
    required this.reason,
  });

  Map<String, dynamic> toJson() => {
        'question_id': questionId,
        'new_score': newScore,
        'reason': reason,
      };
}

class ObjectionInboxItem {
  final String objectionId;
  final String studentId;
  final String questionId;
  final String status;
  final DateTime filedAt;

  const ObjectionInboxItem({
    required this.objectionId,
    required this.studentId,
    required this.questionId,
    required this.status,
    required this.filedAt,
  });

  factory ObjectionInboxItem.fromJson(Map<String, dynamic> json) {
    return ObjectionInboxItem(
      objectionId: json['objection_id'] as String? ?? '',
      studentId: json['student_id'] as String? ?? '',
      questionId: json['question_id'] as String? ?? '',
      status: json['status'] as String? ?? '',
      filedAt: DateTime.tryParse(json['filed_at'] as String? ?? '') ??
          DateTime.now(),
    );
  }
}

// ---------------------------------------------------------------------------
// API client
// ---------------------------------------------------------------------------

class TeacherApi {
  final NetworkService _network;

  TeacherApi(this._network);

  // -- Exams ----------------------------------------------------------------

  Future<List<TeacherExamCard>> listExams({
    String? subjectId,
    String? classId,
  }) async {
    final params = <String, String>{};
    if (subjectId != null) params['subject_id'] = subjectId;
    if (classId != null) params['class_id'] = classId;

    final query =
        params.isNotEmpty ? '?${_encodeQuery(params)}' : '';

    final json = await _network.get<Map<String, dynamic>>(
      '/api/v1/teacher/exams$query',
    );
    final items = json['items'] as List<dynamic>? ?? [];
    return items
        .map((e) => TeacherExamCard.fromJson(e as Map<String, dynamic>))
        .toList(growable: false);
  }

  // -- Scores ---------------------------------------------------------------

  Future<List<ClassScoreRow>> getClassScores(String examId) async {
    final json = await _network.get<Map<String, dynamic>>(
      '/api/v1/teacher/exams/$examId/scores',
    );
    final rows = json['rows'] as List<dynamic>? ?? [];
    return rows
        .map((e) => ClassScoreRow.fromJson(e as Map<String, dynamic>))
        .toList(growable: false);
  }

  Future<TeacherStudentDetail> getStudentDetail(
    String examId,
    String studentId,
  ) async {
    final json = await _network.get<Map<String, dynamic>>(
      '/api/v1/teacher/exams/$examId/students/$studentId',
    );
    return TeacherStudentDetail.fromJson(json);
  }

  Future<Map<String, dynamic>> overrideScore(
    String examId,
    String studentId,
    TeacherScoreOverrideRequest body,
  ) async {
    final json = await _network.patch<Map<String, dynamic>>(
      '/api/v1/teacher/exams/$examId/students/$studentId/questions/${body.questionId}',
      body: body.toJson(),
    );
    return json;
  }

  // -- Objections -----------------------------------------------------------

  Future<List<ObjectionInboxItem>> listObjections(String examId) async {
    final json = await _network.get<Map<String, dynamic>>(
      '/api/v1/teacher/exams/$examId/objections',
    );
    final items = json['items'] as List<dynamic>? ?? [];
    return items
        .map((e) => ObjectionInboxItem.fromJson(e as Map<String, dynamic>))
        .toList(growable: false);
  }

  Future<Map<String, dynamic>> getObjectionDetail(
    String objectionId,
  ) async {
    return await _network.get<Map<String, dynamic>>(
      '/api/v1/teacher/objections/$objectionId',
    );
  }

  Future<Map<String, dynamic>> resolveObjection(
    String objectionId, {
    required String verdict,
    double? newScore,
    required String reason,
  }) async {
    final body = <String, dynamic>{
      'verdict': verdict,
      'reason': reason,
    };
    if (newScore != null) body['new_score'] = newScore;

    return await _network.post<Map<String, dynamic>>(
      '/api/v1/teacher/objections/$objectionId/resolve',
      body: body,
    );
  }

  Future<Map<String, dynamic>> escalateObjection(
    String objectionId, {
    required String targetRole,
    required String reason,
  }) async {
    return await _network.post<Map<String, dynamic>>(
      '/api/v1/teacher/objections/$objectionId/escalate',
      body: {
        'target_role': targetRole,
        'reason': reason,
      },
    );
  }

  // -- Finalize / Publish ---------------------------------------------------

  Future<Map<String, dynamic>> finalizeScores(String examId) async {
    return await _network.post<Map<String, dynamic>>(
      '/api/v1/teacher/exams/$examId/scores/finalize',
      body: {},
    );
  }

  Future<Map<String, dynamic>> publishScores(String examId) async {
    return await _network.post<Map<String, dynamic>>(
      '/api/v1/teacher/exams/$examId/scores/publish',
      body: {},
    );
  }

  // NOTE: The teacher BFF does not expose chat endpoints.
  // Chat is available only through the student BFF.

  // -- Helpers --------------------------------------------------------------

  String _encodeQuery(Map<String, String> params) {
    return params.entries
        .map((e) =>
            '${Uri.encodeComponent(e.key)}=${Uri.encodeComponent(e.value)}')
        .join('&');
  }
}
