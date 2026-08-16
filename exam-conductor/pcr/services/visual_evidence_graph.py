"""Evidence-first contracts for canonical visual PCR grading.

The mapper sees the question paper and the original student pages, but never an
answer key, reference solution, or marking rubric.  The grader receives the
validated mapping plus the immutable marking material.  This separation keeps
question ownership independent from correctness and makes every awarded mark
traceable to an exact region of the submitted copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROMPT_VERSION = "pcr-full-document-visual-v13"
# v13 remains the frozen whole-copy contract.  v14 deliberately uses the same
# normalized evidence graph after validation, but its mapper contract is a
# compact, resumable envelope (no transcription/reason/confidence essays).
V14_PROMPT_VERSION = "pcr-full-document-visual-v14"
V15_PROMPT_VERSION = "pcr-full-document-visual-v15"
COMPACT_MAPPING_VERSION = "pcr-compact-evidence-map-v1"
ORIENTATION_AWARE_MAPPING_VERSION = "pcr-compact-evidence-map-v2"
EVIDENCE_GRAPH_VERSION = "pcr-multimodal-evidence-graph-v3"
COORDINATE_SPACE = "normalized_1000"

_ATTEMPT_STATES = {"attempted", "not_attempted", "unresolved"}
_AUTHORSHIP = {"student", "uncertain"}
_CONTENT_TYPES = {"TEXT_ONLY", "MIXED", "DIAGRAM_HEAVY", "TABLE_PRESENT"}
_EVIDENCE_KINDS = {
    "handwriting",
    "mathematics",
    "diagram",
    "table",
    "graph",
    "label",
    "mixed",
}


@dataclass
class MappingValidationResult:
    document_review: Dict[str, Any]
    questions: Dict[int, Dict[str, Any]]
    unassigned_regions: List[Dict[str, Any]]
    errors: List[str] = field(default_factory=list)

    def as_payload(self) -> Dict[str, Any]:
        return {
            "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
            "document_review": dict(self.document_review),
            "questions": [
                self.questions[number] for number in sorted(self.questions)
            ],
            "unassigned_student_regions": list(self.unassigned_regions),
        }


def strict_provider_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Fail locally when a structured-output schema is not OpenAI-strict.

    Provider strict mode requires every object to close additional properties
    and to list every declared property as required.  Keeping this check beside
    the schema builders prevents a paid request from being the first validator.
    Nullable values must be represented in the property's type rather than by
    omitting that property from ``required``.
    """

    errors: List[str] = []

    def visit(node: Any, path: str) -> None:
        if not isinstance(node, Mapping):
            return
        raw_type = node.get("type")
        types = set(raw_type) if isinstance(raw_type, list) else {raw_type}
        if "object" in types:
            properties = node.get("properties")
            if node.get("additionalProperties") is not False:
                errors.append(f"{path}: object must set additionalProperties=false")
            if not isinstance(properties, Mapping):
                errors.append(f"{path}: object must declare properties")
            else:
                required = node.get("required")
                required_names = set(required) if isinstance(required, list) else set()
                property_names = set(properties)
                if required_names != property_names:
                    errors.append(
                        f"{path}: required properties must exactly match declared properties"
                    )
                for key, value in properties.items():
                    visit(value, f"{path}.properties.{key}")
        items = node.get("items")
        if isinstance(items, Mapping):
            visit(items, f"{path}.items")
        for keyword in ("anyOf", "allOf", "oneOf"):
            variants = node.get(keyword)
            if isinstance(variants, list):
                for index, value in enumerate(variants):
                    visit(value, f"{path}.{keyword}[{index}]")
        definitions = node.get("$defs")
        if isinstance(definitions, Mapping):
            for key, value in definitions.items():
                visit(value, f"{path}.$defs.{key}")

    visit(schema, "schema")
    if errors:
        raise ValueError("Invalid strict structured-output schema: " + "; ".join(errors))
    return schema


def mapping_system_instructions() -> str:
    return (
        "You are the evidence mapper for one complete handwritten answer copy. "
        "This stage never grades and has no answer key, solution, rubric, or marks. "
        "Inspect the original question paper and every original student page directly, "
        "including dark, sideways, angled, faint, multilingual, Hindi, mathematical, "
        "diagram, graph, and table work. Read each page in its natural orientation; "
        "do not depend on OCR.\n\n"
        "Associate all visible STUDENT work with the catalog question it answers. A "
        "question may continue across disconnected regions and distant pages, several "
        "questions may share a page, and answers may be out of order. Use question "
        "numbers, requested task, wording, variables, topic, continuation, and diagram "
        "semantics to establish ownership. Do not judge whether the work is correct.\n\n"
        "Teacher ticks, crosses, circles, corrections, awarded marks, comments, and "
        "model answers are not student evidence. Printed text, headers, names, roll "
        "numbers, page numbers, and administrative marks are also not answer work. "
        "Exclude them. If handwriting authorship itself cannot be separated, mark that "
        "region uncertain instead of treating an annotation as the student's answer.\n\n"
        "Return tight but complete two-dimensional regions in normalized 0..1000 "
        "coordinates relative to the exact original image frame supplied for that page. "
        "Give every region a globally unique stable region_id. Use the same non-empty "
        "continuation_group and increasing sequence for parts of one continued answer. "
        "For diagrams include the complete drawing, labels, arrows, connections, axes, "
        "and nearby explanatory text, and summarize the visible components without "
        "inferring correctness. student_answer must be a faithful concise transcription "
        "or structural description of only the mapped student work; do not correct it.\n\n"
        "Use attempted whenever relevant student work is readable enough to associate. "
        "Use not_attempted only after inspecting every page and finding no work for that "
        "question. Use unresolved only when physical evidence ownership is genuinely "
        "indeterminate; a rotated, dark, faint, or Hindi page is not by itself unresolved. "
        "Put visible student work that cannot be assigned to a question in "
        "unassigned_student_regions. Return every catalog question exactly once."
    )


def grading_system_instructions() -> str:
    return (
        "You are the visual examiner for one complete handwritten answer copy. The "
        "answer mapper has already fixed question ownership. Grade only the supplied "
        "student-authored evidence regions against the immutable question, reference "
        "solution, teacher marking material, and locked criteria. Never remap a region, "
        "borrow work from another question, or use teacher ticks, crosses, comments, "
        "corrections, or written marks as proof of correctness.\n\n"
        "Read the original page images directly in their natural orientation; region "
        "coordinates identify the evidence but do not replace the pixels. Preserve the "
        "student's language and meaning. Evaluate mathematical layout spatially and "
        "evaluate diagrams structurally: labels, arrow endpoints, relationships, axes, "
        "connections, topology, and nearby explanation. The reference solution is a "
        "correctness anchor, not an exact-wording requirement. Equivalent methods, "
        "representations, and wording receive credit.\n\n"
        "Award each locked criterion independently. Give step marks for correct visible "
        "work even when a later step or final answer is wrong, and apply error-carried-"
        "forward only when the locked policy permits it. Visible incorrect or incomplete "
        "work receives the supported partial or zero award; it is not unresolved. Return "
        "every requested criterion exactly once and never exceed its maximum. The server "
        "will derive totals from criterion marks.\n\n"
        "For a catalog question with grading_mode=objective, do not decide correctness. "
        "Return an empty criterion_marks array and total_score 0; the mapper-owned "
        "student_answer already contains the selected label and the server alone applies "
        "the immutable answer key and negative-marking policy.\n\n"
        "For every criterion, cite only region_id values from that question's fixed map. "
        "The evidence text must literally describe what is visible in those cited regions, "
        "including the relevant diagram relationship when applicable. The mapper-owned "
        "student_answer is immutable; do not return or rewrite it. Set needs_review only "
        "when the supplied student evidence is genuinely "
        "indecipherable enough to prevent a reliable score. Do not mention AI, OCR, "
        "confidence, schemas, image processing, or evidence mapping in feedback."
    )


def verification_system_instructions() -> str:
    """Independent, risk-based full-score audit instructions for v14."""

    return (
        "You are an independent second examiner auditing only proposed full-score "
        "subjective answers. You receive the immutable question, rubric, fixed "
        "student evidence map, and original page pixels. You do not receive the "
        "primary examiner's marks or rationale. Independently decide the supported "
        "mark for every locked criterion and cite only region IDs owned by that "
        "question. Do not remap evidence, use teacher annotations, or award marks "
        "without direct visible evidence. Return one row for every criterion, with "
        "a short evidence-based rationale. This is an audit: the server compares "
        "your rows to the provisional result and never changes the score here."
    )


def verification_schema(
    question_contracts: Sequence[Mapping[str, Any]],
    mapping: MappingValidationResult,
) -> Dict[str, Any]:
    """Strict schema for an independent full-score audit batch."""

    question_variants: List[Dict[str, Any]] = []
    for position, contract in enumerate(question_contracts, start=1):
        number = int(contract.get("question_number") or position)
        mapped = mapping.questions.get(number) or {}
        region_ids = [
            str(region.get("region_id") or "")
            for region in mapped.get("evidence_regions") or []
            if str(region.get("region_id") or "")
        ]
        criteria = list(contract.get("marking_criteria") or [])
        criterion_items = []
        for criterion in criteria:
            criterion_id = str(criterion.get("criterion_id") or "")
            region_schema: Dict[str, Any] = {"type": "string"}
            if region_ids:
                region_schema = {"type": "string", "enum": region_ids}
            criterion_items.append({
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "criterion_id": {"type": "string", "enum": [criterion_id]},
                    "marks_supported": {
                        "type": "number", "minimum": 0,
                        "maximum": max(0.0, float(criterion.get("max_marks") or 0)),
                    },
                    "evidence_region_ids": {
                        "type": "array", "items": region_schema,
                        "minItems": 1 if region_ids else 0,
                    },
                    "rationale": {"type": "string", "maxLength": 800},
                },
                "required": ["criterion_id", "marks_supported", "evidence_region_ids", "rationale"],
            })
        criterion_schema: Dict[str, Any] = {
            "type": "object", "additionalProperties": False,
            "properties": {"criterion_id": {"type": "string"}},
            "required": ["criterion_id"],
        }
        if len(criterion_items) == 1:
            criterion_schema = criterion_items[0]
        elif criterion_items:
            criterion_schema = {"anyOf": criterion_items}
        question_variants.append({
            "type": "object", "additionalProperties": False,
            "properties": {
                "question_number": {"type": "integer", "enum": [number]},
                "criterion_marks": {
                    "type": "array", "items": criterion_schema,
                    "minItems": len(criteria), "maxItems": len(criteria),
                },
            },
            "required": ["question_number", "criterion_marks"],
        })
    item_schema: Dict[str, Any] = question_variants[0] if len(question_variants) == 1 else {"anyOf": question_variants}
    return strict_provider_schema({
        "type": "object", "additionalProperties": False,
        "properties": {
            "evidence_graph_version": {"type": "string", "enum": [EVIDENCE_GRAPH_VERSION]},
            "questions": {"type": "array", "items": item_schema, "minItems": len(question_variants), "maxItems": len(question_variants)},
        },
        "required": ["evidence_graph_version", "questions"],
    })


def evidence_region_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "region_id": {"type": "string", "minLength": 1},
            "page_number": {"type": "integer", "minimum": 1},
            "x_start": {"type": "number", "minimum": 0, "maximum": 1000},
            "y_start": {"type": "number", "minimum": 0, "maximum": 1000},
            "x_end": {"type": "number", "minimum": 0, "maximum": 1000},
            "y_end": {"type": "number", "minimum": 0, "maximum": 1000},
            "evidence_kind": {"type": "string", "enum": sorted(_EVIDENCE_KINDS)},
            "authorship": {"type": "string", "enum": sorted(_AUTHORSHIP)},
            "continuation_group": {"type": "string"},
            "sequence": {"type": "integer", "minimum": 1},
            "observed_content": {"type": "string"},
            "diagram_components": {
                "type": "array",
                "items": {"type": "string"},
            },
            "mapping_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "region_id",
            "page_number",
            "x_start",
            "y_start",
            "x_end",
            "y_end",
            "evidence_kind",
            "authorship",
            "continuation_group",
            "sequence",
            "observed_content",
            "diagram_components",
            "mapping_confidence",
        ],
    }


def evidence_mapping_schema(
    question_contracts: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    numbers = [
        int(contract.get("question_number") or index)
        for index, contract in enumerate(question_contracts, start=1)
    ]
    question_variants: List[Dict[str, Any]] = []
    for number in numbers:
        question_variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question_number": {"type": "integer", "enum": [number]},
                    "attempt_status": {
                        "type": "string",
                        "enum": sorted(_ATTEMPT_STATES),
                    },
                    "content_type": {
                        "type": "string",
                        "enum": sorted(_CONTENT_TYPES),
                    },
                    "student_answer": {"type": "string"},
                    "evidence_regions": {
                        "type": "array",
                        "items": evidence_region_schema(),
                    },
                    "mapping_reason": {"type": "string"},
                    "needs_review": {"type": "boolean"},
                    "review_reason": {"type": "string"},
                },
                "required": [
                    "question_number",
                    "attempt_status",
                    "content_type",
                    "student_answer",
                    "evidence_regions",
                    "mapping_reason",
                    "needs_review",
                    "review_reason",
                ],
            }
        )
    question_items: Dict[str, Any]
    if len(question_variants) == 1:
        question_items = question_variants[0]
    else:
        question_items = {"anyOf": question_variants}
    return strict_provider_schema({
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "evidence_graph_version": {
                "type": "string",
                "enum": [EVIDENCE_GRAPH_VERSION],
            },
            "document_review": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "all_student_work_accounted": {"type": "boolean"},
                    "teacher_annotations_present": {"type": "boolean"},
                    "teacher_annotations_excluded": {"type": "boolean"},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "all_student_work_accounted",
                    "teacher_annotations_present",
                    "teacher_annotations_excluded",
                    "warnings",
                ],
            },
            "questions": {
                "type": "array",
                "items": question_items,
                "minItems": len(numbers),
                "maxItems": len(numbers),
            },
            "unassigned_student_regions": {
                "type": "array",
                "items": evidence_region_schema(),
            },
        },
        "required": [
            "evidence_graph_version",
            "document_review",
            "questions",
            "unassigned_student_regions",
        ],
    })


def compact_mapping_system_instructions(
    prompt_version: str = V14_PROMPT_VERSION,
) -> str:
    """Instructions for bounded mapper units.

    This is intentionally short and operational.  The mapper only claims
    ownership and geometry; it does not transcribe, explain, or grade work.
    """

    if prompt_version == V15_PROMPT_VERSION:
        return (
            "You are an answer-key-free student-work ownership mapper. Never grade "
            "and never use a solution, rubric, correctness, teacher ticks, crosses, "
            "circles, corrections, or awarded marks to decide ownership. Inspect every "
            "supplied physical page and associate every visible student-authored answer "
            "block with the catalog question it answers. Several questions may share a "
            "page; one answer may be jumbled or continue across distant pages. Use the "
            "written question number, requested task, wording, topic, variables, answer "
            "structure, continuation, and diagram/table semantics together. Incorrect or "
            "incomplete work is still an attempted answer and must be mapped. Hindi, "
            "multilingual, dark, faint, angled, or sideways work is not by itself "
            "unresolved.\n\n"
            "A physical page may be shown in alternate orientations. These are duplicate "
            "views of the same page, not additional evidence. Use exactly one readable "
            "view for each physical region and return its labelled "
            "source_rotation_degrees_clockwise (0, 90, or 270). Coordinates are tight "
            "normalized_1000 rectangles in that exact displayed view; the server will "
            "invert them into the immutable original-page frame.\n\n"
            "Return attempted for readable student work that can be associated. Return "
            "unresolved only when visible student work cannot be associated after "
            "considering every catalog question. Omit a catalog question only when this "
            "mapping unit contains no visible work for it; whole-copy absence is decided "
            "later by the server. Put genuinely unowned visible student work in "
            "unassigned_student_regions. Do not use unassigned as a shortcut for hard "
            "handwriting. Give a short association_basis and an honest "
            "mapping_confidence. Set all_student_work_accounted=true only when every "
            "visible student-authored region in the supplied physical pages is either "
            "assigned or explicitly returned as unassigned."
        )
    return (
        "You are a bounded student-work ownership mapper. Never grade and never "
        "use an answer key or rubric. Inspect the supplied original page image(s) "
        "and associate visible student work with one catalog question. Answers "
        "may be jumbled, continued on distant pages, multilingual, mathematical, "
        "or diagram/table based. Exclude printed prompts and teacher marks. Return "
        "only compact question ownership and tight normalized_1000 regions. Do not "
        "return transcription, explanations, confidence, or diagram descriptions. "
        "Set all_student_work_accounted=true when every visible student-authored "
        "region on the supplied pages is either assigned to a question or returned "
        "as unassigned; this field does not mean that every catalog question was "
        "attempted, and a blank page can still be fully accounted for. "
        "A returned region must be student-authored; use uncertain authorship when "
        "it cannot safely be separated. Omit questions with no visible work in this "
        "unit. Keep continuation_group and sequence stable across units."
    )


def compact_evidence_region_schema(
    *,
    orientation_aware: bool = False,
    recovery_pass: bool = False,
) -> Dict[str, Any]:
    properties: Dict[str, Any] = {
        "region_id": {"type": "string", "minLength": 1, "maxLength": 120},
        "page_number": {"type": "integer", "minimum": 1},
        "x_start": {"type": "number", "minimum": 0, "maximum": 1000},
        "y_start": {"type": "number", "minimum": 0, "maximum": 1000},
        "x_end": {"type": "number", "minimum": 0, "maximum": 1000},
        "y_end": {"type": "number", "minimum": 0, "maximum": 1000},
        "evidence_kind": {"type": "string", "enum": sorted(_EVIDENCE_KINDS)},
        "authorship": {"type": "string", "enum": sorted(_AUTHORSHIP)},
        "continuation_group": {"type": "string", "maxLength": 120},
        "sequence": {"type": "integer", "minimum": 1},
    }
    if orientation_aware:
        properties.update({
            "source_rotation_degrees_clockwise": {
                "type": "integer",
                "enum": [0, 90, 270],
            },
            "mapping_confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        })
    if recovery_pass:
        properties["supersedes_region_ids"] = {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 120},
            "maxItems": 16,
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def compact_mapping_schema(
    question_contracts: Sequence[Mapping[str, Any]],
    *,
    prompt_version: str = V14_PROMPT_VERSION,
    recovery_pass: bool = False,
) -> Dict[str, Any]:
    """Schema for one bounded mapper unit.

    Unlike the v13 schema, questions are optional and regions contain no prose;
    this makes output size proportional to visible ownership, not paper size.
    """

    numbers = [
        int(contract.get("question_number") or index)
        for index, contract in enumerate(question_contracts, start=1)
    ]
    orientation_aware = prompt_version == V15_PROMPT_VERSION
    question_properties: Dict[str, Any] = {
        "question_number": {"type": "integer", "enum": numbers},
        "attempt_status": {"type": "string", "enum": ["attempted", "unresolved"]},
        "content_type": {"type": "string", "enum": sorted(_CONTENT_TYPES)},
        "evidence_regions": {
            "type": "array",
            "items": compact_evidence_region_schema(
                orientation_aware=orientation_aware,
                recovery_pass=recovery_pass,
            ),
            "maxItems": 64,
        },
    }
    if orientation_aware:
        question_properties["association_basis"] = {
            "type": "string",
            "maxLength": 240,
        }
    question = {
        "type": "object",
        "additionalProperties": False,
        "properties": question_properties,
        "required": list(question_properties),
    }
    return strict_provider_schema({
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mapping_version": {
                "type": "string",
                "enum": [
                    ORIENTATION_AWARE_MAPPING_VERSION
                    if orientation_aware
                    else COMPACT_MAPPING_VERSION
                ],
            },
            "evidence_graph_version": {"type": "string", "enum": [EVIDENCE_GRAPH_VERSION]},
            "all_student_work_accounted": {"type": "boolean"},
            "questions": {"type": "array", "items": question, "maxItems": len(numbers)},
            "unassigned_student_regions": {
                "type": "array",
                "items": compact_evidence_region_schema(
                    orientation_aware=orientation_aware,
                    recovery_pass=recovery_pass,
                ),
                "maxItems": 64,
            },
        },
        "required": ["mapping_version", "evidence_graph_version", "all_student_work_accounted", "questions", "unassigned_student_regions"],
    })


def normalize_compact_mapping_payload(
    payload: Any,
    *,
    question_numbers: Sequence[int],
    page_count: int,
) -> MappingValidationResult:
    """Normalize a compact unit aggregate into the existing graph validator."""

    raw = payload if isinstance(payload, Mapping) else {}
    expanded_questions = []
    for item in raw.get("questions") or []:
        if not isinstance(item, Mapping):
            continue
        expanded_questions.append({
            "question_number": item.get("question_number"),
            "attempt_status": item.get("attempt_status", "attempted"),
            "content_type": item.get("content_type", "MIXED"),
            "student_answer": "",
            "evidence_regions": [
                {
                    **dict(region),
                    "observed_content": "",
                    "diagram_components": [],
                    "mapping_confidence": region.get("mapping_confidence", 1.0),
                }
                for region in (item.get("evidence_regions") or [])
                if isinstance(region, Mapping)
            ],
            "mapping_reason": str(item.get("association_basis") or "")[:240],
            "needs_review": False,
            "review_reason": "",
        })
    expanded = {
        "evidence_graph_version": raw.get("evidence_graph_version", EVIDENCE_GRAPH_VERSION),
        "document_review": {
            "all_student_work_accounted": bool(raw.get("all_student_work_accounted")),
            "teacher_annotations_present": False,
            "teacher_annotations_excluded": True,
            "warnings": [],
        },
        "questions": expanded_questions,
        "unassigned_student_regions": [
            {
                **dict(region),
                "observed_content": "",
                "diagram_components": [],
                "mapping_confidence": region.get("mapping_confidence", 1.0),
            }
            for region in (raw.get("unassigned_student_regions") or [])
            if isinstance(region, Mapping)
        ],
    }
    return validate_mapping_payload(
        expanded, question_numbers=question_numbers, page_count=page_count
    )


def merge_compact_mapping_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    question_numbers: Sequence[int],
    page_count: int,
) -> MappingValidationResult:
    """Globally merge bounded page/question units before grading.

    Region IDs are immutable ownership atoms.  Duplicate atoms are rejected by
    the normal validator and overlapping claims become unresolved, so a retry
    can never silently award work twice.
    """

    questions: Dict[int, Dict[str, Any]] = {}
    unassigned: List[Dict[str, Any]] = []
    complete = True
    for payload in payloads:
        raw = payload if isinstance(payload, Mapping) else {}
        complete = complete and bool(raw.get("all_student_work_accounted"))
        for item in raw.get("questions") or []:
            if not isinstance(item, Mapping):
                continue
            number = _positive_int(item.get("question_number"))
            if not number:
                continue
            target = questions.setdefault(number, {
                "question_number": number,
                "attempt_status": "attempted",
                "content_type": item.get("content_type") or "MIXED",
                "association_basis": str(item.get("association_basis") or "")[:240],
                "evidence_regions": [],
            })
            target["evidence_regions"].extend(
                region for region in (item.get("evidence_regions") or []) if isinstance(region, Mapping)
            )
            if str(item.get("attempt_status") or "") == "unresolved":
                target["attempt_status"] = "unresolved"
            if not target.get("association_basis") and item.get("association_basis"):
                target["association_basis"] = str(item.get("association_basis"))[:240]
        unassigned.extend(
            region for region in (raw.get("unassigned_student_regions") or [])
            if isinstance(region, Mapping)
        )
    # A complete, globally accounted copy proves absence only when there is no
    # unassigned work.  Canonicalize continuation labels because independent
    # page units cannot be expected to invent the same opaque label.
    can_prove_absence = complete and not unassigned
    for number, item in questions.items():
        regions = list(item.get("evidence_regions") or [])
        if len(regions) > 1:
            ordered = sorted(
                regions,
                key=lambda region: (
                    int(region.get("page_number") or 0),
                    int(region.get("sequence") or 1),
                    str(region.get("region_id") or ""),
                ),
            )
            for sequence, region in enumerate(ordered, start=1):
                region["continuation_group"] = f"q{number}-answer"
                region["sequence"] = sequence
            item["evidence_regions"] = ordered
    if can_prove_absence:
        for number in question_numbers:
            if number not in questions:
                questions[number] = {
                    "question_number": number,
                    "attempt_status": "not_attempted",
                    "content_type": "TEXT_ONLY",
                    "evidence_regions": [],
                }
    aggregate = {
        "mapping_version": (
            ORIENTATION_AWARE_MAPPING_VERSION
            if any(
                str((payload or {}).get("mapping_version") or "")
                == ORIENTATION_AWARE_MAPPING_VERSION
                for payload in payloads
                if isinstance(payload, Mapping)
            )
            else COMPACT_MAPPING_VERSION
        ),
        "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
        "all_student_work_accounted": complete,
        "questions": list(questions.values()),
        "unassigned_student_regions": unassigned,
    }
    return normalize_compact_mapping_payload(
        aggregate, question_numbers=question_numbers, page_count=page_count
    )


def reconcile_compact_mapping_recovery(
    initial_payloads: Sequence[Mapping[str, Any]],
    recovery_payloads: Sequence[Mapping[str, Any]],
    *,
    question_numbers: Sequence[int],
    page_count: int,
    minimum_coverage: float = 0.6,
    recovered_page_numbers: Optional[Sequence[int]] = None,
) -> MappingValidationResult:
    """Merge one bounded recovery pass without erasing unexplained work.

    Initial unassigned regions are removed only when the recovery output covers
    them geometrically with assigned or still-unassigned regions.  A sparse or
    overconfident recovery response therefore cannot turn visible work into a
    false ``not_attempted`` zero.
    """

    initial_regions_by_id: Dict[str, Dict[str, Any]] = {}
    initial_assigned_regions: List[Dict[str, Any]] = []
    for payload in initial_payloads:
        if not isinstance(payload, Mapping):
            continue
        for question in payload.get("questions") or []:
            if not isinstance(question, Mapping):
                continue
            for region in question.get("evidence_regions") or []:
                if not isinstance(region, Mapping):
                    continue
                item = dict(region)
                region_id = str(item.get("region_id") or "")
                if region_id:
                    initial_regions_by_id[region_id] = item
                initial_assigned_regions.append(item)
        for region in payload.get("unassigned_student_regions") or []:
            if not isinstance(region, Mapping):
                continue
            item = dict(region)
            region_id = str(item.get("region_id") or "")
            if region_id:
                initial_regions_by_id[region_id] = item

    supersede_counts: Dict[str, int] = {}
    for payload in recovery_payloads:
        if not isinstance(payload, Mapping):
            continue
        for question in payload.get("questions") or []:
            if not isinstance(question, Mapping):
                continue
            for region in question.get("evidence_regions") or []:
                if not isinstance(region, Mapping):
                    continue
                for region_id in region.get("supersedes_region_ids") or []:
                    key = str(region_id or "")
                    if key:
                        supersede_counts[key] = supersede_counts.get(key, 0) + 1

    valid_superseded_ids: set[str] = set()
    sanitized_recovery: List[Dict[str, Any]] = []
    for payload in recovery_payloads:
        if not isinstance(payload, Mapping):
            continue
        sanitized = dict(payload)
        sanitized_questions: List[Dict[str, Any]] = []
        invalid_regions: List[Dict[str, Any]] = []
        for question in payload.get("questions") or []:
            if not isinstance(question, Mapping):
                continue
            sanitized_question = dict(question)
            kept_regions: List[Dict[str, Any]] = []
            for region in question.get("evidence_regions") or []:
                if not isinstance(region, Mapping):
                    continue
                candidate = dict(region)
                supersedes = [
                    str(value) for value in candidate.get("supersedes_region_ids") or []
                    if str(value)
                ]
                valid = all(
                    supersede_counts.get(region_id) == 1
                    and region_id in initial_regions_by_id
                    and _regions_materially_overlap(
                        initial_regions_by_id[region_id],
                        candidate,
                        minimum_overlap=minimum_coverage,
                    )
                    for region_id in supersedes
                )
                overlapping_assigned_ids = {
                    str(source.get("region_id") or "")
                    for source in initial_assigned_regions
                    if _regions_materially_overlap(
                        source,
                        candidate,
                        minimum_overlap=minimum_coverage,
                    )
                }
                # Recovery may discover a new region, but it cannot silently
                # duplicate or steal an already assigned region. Ownership
                # changes require an exact, geometrically supported supersede.
                if overlapping_assigned_ids - set(supersedes):
                    valid = False
                if supersedes and valid:
                    valid_superseded_ids.update(supersedes)
                if valid:
                    kept_regions.append(candidate)
                else:
                    candidate["authorship"] = "uncertain"
                    invalid_regions.append(candidate)
            sanitized_question["evidence_regions"] = kept_regions
            if kept_regions:
                sanitized_questions.append(sanitized_question)
        sanitized["questions"] = sanitized_questions
        sanitized["unassigned_student_regions"] = [
            dict(region)
            for region in payload.get("unassigned_student_regions") or []
            if isinstance(region, Mapping)
        ] + invalid_regions
        sanitized_recovery.append(sanitized)

    recovery_regions = [
        dict(region)
        for payload in sanitized_recovery
        for question in (payload.get("questions") or [])
        if isinstance(question, Mapping)
        for region in (question.get("evidence_regions") or [])
        if isinstance(region, Mapping)
    ]
    recovery_regions.extend(
        dict(region)
        for payload in sanitized_recovery
        for region in (payload.get("unassigned_student_regions") or [])
        if isinstance(region, Mapping)
    )

    recovery_proves_document_complete = (
        set(int(page) for page in (recovered_page_numbers or []) if int(page) > 0)
        == set(range(1, max(1, int(page_count)) + 1))
        and bool(recovery_payloads)
        and all(
            bool(payload.get("all_student_work_accounted"))
            for payload in recovery_payloads
            if isinstance(payload, Mapping)
        )
    )
    adjusted_initial: List[Dict[str, Any]] = []
    for payload in initial_payloads:
        if not isinstance(payload, Mapping):
            continue
        adjusted = dict(payload)
        if recovery_proves_document_complete:
            # The bounded recovery pass re-inspected every physical page, so
            # its complete coverage supersedes an earlier incomplete flag.
            adjusted["all_student_work_accounted"] = True
        adjusted_questions: List[Dict[str, Any]] = []
        for item in payload.get("questions") or []:
            if not isinstance(item, Mapping):
                continue
            adjusted_question = dict(item)
            adjusted_question["evidence_regions"] = [
                dict(region)
                for region in item.get("evidence_regions") or []
                if isinstance(region, Mapping)
                and str(region.get("region_id") or "") not in valid_superseded_ids
            ]
            if adjusted_question["evidence_regions"]:
                adjusted_questions.append(adjusted_question)
        adjusted["questions"] = adjusted_questions
        adjusted["unassigned_student_regions"] = [
            dict(region)
            for region in (payload.get("unassigned_student_regions") or [])
            if isinstance(region, Mapping)
            and not _region_covered_by_recovery(
                region,
                recovery_regions,
                minimum_coverage=minimum_coverage,
            )
        ]
        adjusted_initial.append(adjusted)

    return merge_compact_mapping_payloads(
        [*adjusted_initial, *sanitized_recovery],
        question_numbers=question_numbers,
        page_count=page_count,
    )


def _regions_materially_overlap(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    minimum_overlap: float,
) -> bool:
    if _positive_int(first.get("page_number")) != _positive_int(second.get("page_number")):
        return False
    first_values = {
        key: _finite_float(first.get(key))
        for key in ("x_start", "y_start", "x_end", "y_end")
    }
    second_values = {
        key: _finite_float(second.get(key))
        for key in ("x_start", "y_start", "x_end", "y_end")
    }
    if any(value is None for value in (*first_values.values(), *second_values.values())):
        return False
    intersection_width = max(
        0.0,
        min(float(first_values["x_end"]), float(second_values["x_end"]))
        - max(float(first_values["x_start"]), float(second_values["x_start"])),
    )
    intersection_height = max(
        0.0,
        min(float(first_values["y_end"]), float(second_values["y_end"]))
        - max(float(first_values["y_start"]), float(second_values["y_start"])),
    )
    first_area = max(0.0, float(first_values["x_end"]) - float(first_values["x_start"])) * max(
        0.0, float(first_values["y_end"]) - float(first_values["y_start"])
    )
    second_area = max(0.0, float(second_values["x_end"]) - float(second_values["x_start"])) * max(
        0.0, float(second_values["y_end"]) - float(second_values["y_start"])
    )
    smaller_area = min(first_area, second_area)
    if smaller_area <= 0:
        return False
    ratio = intersection_width * intersection_height / smaller_area
    return ratio >= max(0.0, min(1.0, float(minimum_overlap)))


def _region_covered_by_recovery(
    source: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    minimum_coverage: float,
) -> bool:
    page = _positive_int(source.get("page_number"))
    values = {
        key: _finite_float(source.get(key))
        for key in ("x_start", "y_start", "x_end", "y_end")
    }
    if not page or any(value is None for value in values.values()):
        return False
    source_area = max(0.0, float(values["x_end"]) - float(values["x_start"])) * max(
        0.0, float(values["y_end"]) - float(values["y_start"])
    )
    if source_area <= 0:
        return False
    covered_area = 0.0
    for candidate in candidates:
        if _positive_int(candidate.get("page_number")) != page:
            continue
        candidate_values = {
            key: _finite_float(candidate.get(key))
            for key in ("x_start", "y_start", "x_end", "y_end")
        }
        if any(value is None for value in candidate_values.values()):
            continue
        width = max(
            0.0,
            min(float(values["x_end"]), float(candidate_values["x_end"]))
            - max(float(values["x_start"]), float(candidate_values["x_start"])),
        )
        height = max(
            0.0,
            min(float(values["y_end"]), float(candidate_values["y_end"]))
            - max(float(values["y_start"]), float(candidate_values["y_start"])),
        )
        covered_area += width * height
    return min(1.0, covered_area / source_area) >= max(
        0.0, min(1.0, float(minimum_coverage))
    )


def grading_schema(
    question_contracts: Sequence[Mapping[str, Any]],
    mapping: MappingValidationResult,
) -> Dict[str, Any]:
    variants = []
    for position, contract in enumerate(question_contracts, start=1):
        number = int(contract.get("question_number") or position)
        mapped = mapping.questions.get(number) or {}
        region_ids = [
            str(region.get("region_id") or "")
            for region in mapped.get("evidence_regions") or []
            if str(region.get("region_id") or "")
        ]
        criteria = list(contract.get("marking_criteria") or [])
        criterion_variants = [
            _criterion_schema(criterion, region_ids) for criterion in criteria
        ]
        criterion_items: Dict[str, Any]
        if len(criterion_variants) == 1:
            criterion_items = criterion_variants[0]
        elif criterion_variants:
            criterion_items = {"anyOf": criterion_variants}
        else:
            criterion_items = _criterion_schema(None, region_ids)
        variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question_number": {"type": "integer", "enum": [number]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "criterion_marks": {
                        "type": "array",
                        "items": criterion_items,
                        "minItems": len(criteria),
                        "maxItems": len(criteria),
                    },
                    "total_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": max(0.0, float(contract.get("max_marks") or 0)),
                    },
                    "overall_feedback": {"type": "string"},
                    "needs_review": {"type": "boolean"},
                    "review_reason": {"type": "string"},
                },
                "required": [
                    "question_number",
                    "confidence",
                    "criterion_marks",
                    "total_score",
                    "overall_feedback",
                    "needs_review",
                    "review_reason",
                ],
            }
        )
    items: Dict[str, Any]
    if len(variants) == 1:
        items = variants[0]
    else:
        items = {"anyOf": variants}
    return strict_provider_schema({
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "evidence_graph_version": {
                "type": "string",
                "enum": [EVIDENCE_GRAPH_VERSION],
            },
            "questions": {
                "type": "array",
                "items": items,
                "minItems": len(variants),
                "maxItems": len(variants),
            },
        },
        "required": ["evidence_graph_version", "questions"],
    })


def _criterion_schema(
    criterion: Optional[Mapping[str, Any]],
    region_ids: Sequence[str],
) -> Dict[str, Any]:
    id_schema: Dict[str, Any] = {"type": "string", "minLength": 1}
    max_marks = 0.0
    if criterion is not None:
        criterion_id = str(criterion.get("criterion_id") or "")
        id_schema = {"type": "string", "enum": [criterion_id]}
        max_marks = max(0.0, float(criterion.get("max_marks") or 0))
    region_id_schema: Dict[str, Any] = {"type": "string"}
    if region_ids:
        region_id_schema = {"type": "string", "enum": list(region_ids)}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "criterion_id": id_schema,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "marks_awarded": {
                "type": "number",
                "minimum": 0,
                "maximum": max_marks,
            },
            "rationale": {"type": "string"},
            "evidence": {"type": "string"},
            "evidence_region_ids": {
                "type": "array",
                "items": region_id_schema,
                "minItems": 1 if region_ids else 0,
            },
            "credit_basis": {
                "type": "string",
                "enum": ["direct_evidence", "error_carried_forward", "no_credit"],
            },
        },
        "required": [
            "criterion_id",
            "confidence",
            "marks_awarded",
            "rationale",
            "evidence",
            "evidence_region_ids",
            "credit_basis",
        ],
    }


def validate_mapping_payload(
    payload: Any,
    *,
    question_numbers: Sequence[int],
    page_count: int,
) -> MappingValidationResult:
    raw = payload if isinstance(payload, Mapping) else {}
    errors: List[str] = []
    if raw.get("evidence_graph_version") != EVIDENCE_GRAPH_VERSION:
        errors.append("Evidence graph version does not match the required contract")

    review_raw = raw.get("document_review")
    if not isinstance(review_raw, Mapping):
        review_raw = {}
        errors.append("Document evidence review is missing")
    review = {
        "all_student_work_accounted": bool(
            review_raw.get("all_student_work_accounted")
        ),
        "teacher_annotations_present": bool(
            review_raw.get("teacher_annotations_present")
        ),
        "teacher_annotations_excluded": bool(
            review_raw.get("teacher_annotations_excluded")
        ),
        "warnings": [
            str(item).strip()[:500]
            for item in (review_raw.get("warnings") or [])
            if str(item).strip()
        ],
    }
    if review["teacher_annotations_present"] and not review[
        "teacher_annotations_excluded"
    ]:
        errors.append("Teacher annotations were not safely separated from student work")

    expected = {int(number) for number in question_numbers}
    questions: Dict[int, Dict[str, Any]] = {}
    global_region_ids: set[str] = set()
    raw_questions = raw.get("questions")
    if not isinstance(raw_questions, list):
        raw_questions = []
        errors.append("Question evidence mappings are missing")
    for item in raw_questions:
        if not isinstance(item, Mapping):
            errors.append("Question evidence mapping is not an object")
            continue
        number = _positive_int(item.get("question_number"))
        if number not in expected:
            errors.append("Evidence mapping refers to an unknown question")
            continue
        if number in questions:
            errors.append(f"Q{number} has duplicate evidence mappings")
            questions[number] = _unresolved_mapping(
                number, "Duplicate question evidence mappings require review"
            )
            continue
        questions[number] = _validate_question_mapping(
            item,
            question_number=number,
            page_count=page_count,
            document_complete=review["all_student_work_accounted"],
            global_region_ids=global_region_ids,
            errors=errors,
        )

    for number in sorted(expected - set(questions)):
        reason = "No evidence mapping was returned for this question"
        errors.append(f"Q{number}: {reason}")
        questions[number] = _unresolved_mapping(number, reason)

    conflicting_questions: set[int] = set()
    owned_regions = [
        (number, region)
        for number, question in questions.items()
        for region in (question.get("evidence_regions") or [])
    ]
    for index, (left_number, left_region) in enumerate(owned_regions):
        for right_number, right_region in owned_regions[index + 1 :]:
            if left_number == right_number:
                continue
            if _region_overlap_fraction(left_region, right_region) < 0.85:
                continue
            conflicting_questions.update({left_number, right_number})
            errors.append(
                f"Q{left_number} and Q{right_number} claim substantially the same "
                "student evidence region"
            )
    for number in conflicting_questions:
        question = questions[number]
        question["attempt_status"] = "unresolved"
        question["needs_review"] = True
        question["review_reason"] = (
            "The same physical student work was assigned to more than one question"
        )

    unassigned: List[Dict[str, Any]] = []
    raw_unassigned = raw.get("unassigned_student_regions")
    if not isinstance(raw_unassigned, list):
        raw_unassigned = []
        errors.append("Unassigned student regions must be an array")
    for index, item in enumerate(raw_unassigned, start=1):
        region, region_errors = _validate_region(
            item,
            page_count=page_count,
            fallback_region_id=f"unassigned-{index}",
        )
        errors.extend(f"Unassigned region: {error}" for error in region_errors)
        if region:
            region_id = region["region_id"]
            if region_id in global_region_ids:
                errors.append(f"Duplicate global evidence region ID {region_id}")
            else:
                global_region_ids.add(region_id)
                unassigned.append(region)
    if unassigned:
        review["all_student_work_accounted"] = False
        review["warnings"].append(
            f"{len(unassigned)} visible student-work region(s) remain unassigned"
        )
    if errors:
        review["all_student_work_accounted"] = False
        review["warnings"].append(
            "The evidence graph has structural or authorship validation errors"
        )
    review["warnings"] = list(dict.fromkeys(review["warnings"]))
    return MappingValidationResult(review, questions, unassigned, errors)


def _validate_question_mapping(
    raw: Mapping[str, Any],
    *,
    question_number: int,
    page_count: int,
    document_complete: bool,
    global_region_ids: set[str],
    errors: List[str],
) -> Dict[str, Any]:
    status = str(raw.get("attempt_status") or "unresolved").strip().lower()
    if status not in _ATTEMPT_STATES:
        status = "unresolved"
    regions: List[Dict[str, Any]] = []
    for index, item in enumerate(raw.get("evidence_regions") or [], start=1):
        region, region_errors = _validate_region(
            item,
            page_count=page_count,
            fallback_region_id=f"q{question_number}-region-{index}",
        )
        errors.extend(f"Q{question_number}: {error}" for error in region_errors)
        if not region:
            continue
        region_id = region["region_id"]
        if region_id in global_region_ids:
            errors.append(f"Q{question_number}: duplicate global region ID {region_id}")
            continue
        global_region_ids.add(region_id)
        regions.append(region)

    needs_review = bool(raw.get("needs_review"))
    reason = str(raw.get("review_reason") or "").strip()
    if any(region.get("authorship") != "student" for region in regions):
        status = "unresolved"
        needs_review = True
        reason = reason or "Student authorship is uncertain for mapped evidence"
    if status == "attempted" and not regions:
        status = "unresolved"
        needs_review = True
        reason = reason or "Attempted answer has no valid student evidence region"
    if status == "not_attempted":
        if regions or not document_complete:
            status = "unresolved"
            needs_review = True
            reason = reason or "The complete-copy mapping did not prove this answer absent"
    if status == "unresolved":
        needs_review = True
        reason = reason or "Question evidence ownership is unresolved"
    content_type = str(raw.get("content_type") or "MIXED").strip().upper()
    if content_type not in _CONTENT_TYPES:
        content_type = "MIXED"
    return {
        "question_number": question_number,
        "attempt_status": status,
        "confidence": min(
            [float(region.get("mapping_confidence") or 0) for region in regions]
            or [1.0 if status == "not_attempted" else 0.0]
        ),
        "content_type": content_type,
        "student_answer": str(raw.get("student_answer") or "").strip()[:8000],
        "evidence_regions": sorted(
            regions,
            key=lambda item: (
                str(item.get("continuation_group") or ""),
                int(item.get("sequence") or 1),
                int(item.get("page_number") or 0),
            ),
        ),
        "mapping_reason": str(raw.get("mapping_reason") or "").strip()[:1000],
        "needs_review": needs_review,
        "review_reason": reason[:800],
    }


def _validate_region(
    raw: Any,
    *,
    page_count: int,
    fallback_region_id: str,
) -> tuple[Optional[Dict[str, Any]], List[str]]:
    if not isinstance(raw, Mapping):
        return None, ["Evidence region is not an object"]
    page_number = _positive_int(raw.get("page_number"))
    if not page_number or page_number > page_count:
        return None, ["Evidence refers to a non-submitted page"]
    values = {key: _finite_float(raw.get(key)) for key in (
        "x_start", "y_start", "x_end", "y_end"
    )}
    if any(value is None for value in values.values()):
        return None, ["Evidence region is missing two-dimensional coordinates"]
    x_start = float(values["x_start"])
    y_start = float(values["y_start"])
    x_end = float(values["x_end"])
    y_end = float(values["y_end"])
    if (
        x_start < 0
        or y_start < 0
        or x_end > 1000
        or y_end > 1000
        or x_end <= x_start
        or y_end <= y_start
    ):
        return None, ["Evidence has an invalid two-dimensional page region"]
    errors: List[str] = []
    region_id = str(raw.get("region_id") or fallback_region_id).strip()[:120]
    kind = str(raw.get("evidence_kind") or "mixed").strip().lower()
    if kind not in _EVIDENCE_KINDS:
        kind = "mixed"
        errors.append("Evidence kind was normalized to mixed")
    authorship = str(raw.get("authorship") or "uncertain").strip().lower()
    if authorship not in _AUTHORSHIP:
        authorship = "uncertain"
        errors.append("Evidence authorship is invalid")
    return (
        {
            "region_id": region_id,
            "page_number": page_number,
            "x_start": round(x_start, 3),
            "y_start": round(y_start, 3),
            "x_end": round(x_end, 3),
            "y_end": round(y_end, 3),
            "coordinate_space": COORDINATE_SPACE,
            "coordinate_frame": dict(raw.get("coordinate_frame"))
            if isinstance(raw.get("coordinate_frame"), Mapping)
            else raw.get("coordinate_frame"),
            "evidence_kind": kind,
            "authorship": authorship,
            "continuation_group": str(raw.get("continuation_group") or "").strip()[:120],
            "sequence": max(1, int(raw.get("sequence") or 1)),
            "evidence": str(raw.get("observed_content") or "").strip()[:1000],
            "diagram_components": [
                str(item).strip()[:300]
                for item in (raw.get("diagram_components") or [])
                if str(item).strip()
            ][:30],
            "mapping_confidence": _confidence(raw.get("mapping_confidence")),
        },
        errors,
    )


def merge_mapping_and_grading(
    mapping: MappingValidationResult,
    grading_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_grades = grading_payload.get("questions")
    grade_by_number: Dict[int, Dict[str, Any]] = {}
    if isinstance(raw_grades, list):
        for item in raw_grades:
            if not isinstance(item, Mapping):
                continue
            number = _positive_int(item.get("question_number"))
            if number and number not in grade_by_number:
                grade_by_number[number] = dict(item)

    questions: List[Dict[str, Any]] = []
    for number in sorted(mapping.questions):
        mapped = mapping.questions[number]
        status = str(mapped.get("attempt_status") or "unresolved")
        common = {
            "question_number": number,
            "attempt_status": status,
            "confidence": _confidence(mapped.get("confidence")),
            "content_type": mapped.get("content_type") or "MIXED",
            "student_answer": str(mapped.get("student_answer") or "").strip(),
            "source_pages": list(mapped.get("evidence_regions") or []),
            "mapping_reason": mapped.get("mapping_reason") or "",
        }
        if status == "not_attempted":
            questions.append(
                {
                    **common,
                    "student_answer": "",
                    "criterion_marks": [],
                    "total_score": 0.0,
                    "overall_feedback": "Question not attempted.",
                    "needs_review": False,
                    "review_reason": "",
                }
            )
            continue
        if status == "unresolved":
            questions.append(
                {
                    **common,
                    "student_answer": common["student_answer"],
                    "criterion_marks": [],
                    "total_score": 0.0,
                    "overall_feedback": "Answer evidence requires review.",
                    "needs_review": True,
                    "review_reason": mapped.get("review_reason")
                    or "Question evidence ownership is unresolved",
                }
            )
            continue
        grade = grade_by_number.get(number)
        if grade is None:
            questions.append(
                {
                    **common,
                    "attempt_status": "unresolved",
                    "student_answer": common["student_answer"],
                    "criterion_marks": [],
                    "total_score": 0.0,
                    "overall_feedback": "Visual grading did not complete.",
                    "needs_review": True,
                    "review_reason": "The grading result is missing for mapped evidence",
                }
            )
            continue
        display_answer = common["student_answer"]
        if not display_answer:
            evidence_fragments = []
            for criterion in grade.get("criterion_marks") or []:
                if not isinstance(criterion, Mapping):
                    continue
                fragment = str(criterion.get("evidence") or "").strip()
                if fragment and fragment not in evidence_fragments:
                    evidence_fragments.append(fragment)
            display_answer = " ".join(evidence_fragments)[:1200]
        questions.append(
            {
                **common,
                "student_answer": display_answer,
                "confidence": min(
                    _confidence(mapped.get("confidence")),
                    _confidence(grade.get("confidence")),
                ),
                "criterion_marks": list(grade.get("criterion_marks") or []),
                "total_score": grade.get("total_score", 0.0),
                "overall_feedback": str(grade.get("overall_feedback") or "").strip(),
                "needs_review": bool(
                    mapped.get("needs_review") or grade.get("needs_review")
                ),
                "review_reason": str(
                    mapped.get("review_reason") or grade.get("review_reason") or ""
                ).strip(),
            }
        )
    return {
        "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
        "document_review": dict(mapping.document_review),
        "validation_errors": list(mapping.errors),
        "questions": questions,
        "unassigned_student_regions": list(mapping.unassigned_regions),
    }


def _unresolved_mapping(question_number: int, reason: str) -> Dict[str, Any]:
    return {
        "question_number": question_number,
        "attempt_status": "unresolved",
        "confidence": 0.0,
        "content_type": "MIXED",
        "student_answer": "",
        "evidence_regions": [],
        "mapping_reason": "",
        "needs_review": True,
        "review_reason": reason,
    }


def _region_overlap_fraction(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> float:
    if int(left.get("page_number") or 0) != int(right.get("page_number") or 0):
        return 0.0
    left_width = max(
        0.0, float(left.get("x_end") or 0) - float(left.get("x_start") or 0)
    )
    left_height = max(
        0.0, float(left.get("y_end") or 0) - float(left.get("y_start") or 0)
    )
    right_width = max(
        0.0, float(right.get("x_end") or 0) - float(right.get("x_start") or 0)
    )
    right_height = max(
        0.0, float(right.get("y_end") or 0) - float(right.get("y_start") or 0)
    )
    smaller_area = min(left_width * left_height, right_width * right_height)
    if smaller_area <= 0:
        return 0.0
    intersection_width = max(
        0.0,
        min(float(left.get("x_end") or 0), float(right.get("x_end") or 0))
        - max(float(left.get("x_start") or 0), float(right.get("x_start") or 0)),
    )
    intersection_height = max(
        0.0,
        min(float(left.get("y_end") or 0), float(right.get("y_end") or 0))
        - max(float(left.get("y_start") or 0), float(right.get("y_start") or 0)),
    )
    return intersection_width * intersection_height / smaller_area


def _positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _confidence(value: Any) -> float:
    parsed = _finite_float(value)
    if parsed is None:
        return 0.0
    return max(0.0, min(1.0, parsed))
