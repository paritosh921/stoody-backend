"""Deterministic layout inspection for manual-region OCR crops."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


OPTION_LABEL_RE = re.compile(r"^\s*[\(\[]?([a-dA-D])[\.\)]\s*(.*)$")


def group_text_items_into_lines(
    text_items: Iterable[Dict[str, Any]],
    *,
    y_tolerance: float = 4.0,
) -> List[Dict[str, Any]]:
    """Group PDF text items into visual lines using nearby y coordinates."""
    normalized: List[Dict[str, Any]] = []
    for item in text_items or []:
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        normalized.append(
            {
                "text": text,
                "x": float(item.get("x", 0) or 0),
                "y": float(item.get("y", 0) or 0),
                "width": float(item.get("width", 0) or 0),
                "height": float(item.get("height", 0) or 0),
            }
        )

    lines: List[Dict[str, Any]] = []
    for item in sorted(normalized, key=lambda i: (i["y"], i["x"])):
        target: Optional[Dict[str, Any]] = None
        for line in lines:
            if abs(line["y"] - item["y"]) <= y_tolerance:
                target = line
                break
        if target is None:
            target = {"items": [], "x": item["x"], "y": item["y"], "height": item["height"]}
            lines.append(target)
        target["items"].append(item)
        target["x"] = min(target["x"], item["x"])
        target["y"] = min(target["y"], item["y"])
        target["height"] = max(target["height"], item["height"])

    for line in lines:
        line["items"].sort(key=lambda i: i["x"])
        line["text"] = " ".join(i["text"] for i in line["items"]).strip()
        match = OPTION_LABEL_RE.match(line["text"])
        if match:
            line["option_label"] = match.group(1).lower()
            line["option_text"] = match.group(2).strip()
            line["label_only"] = not bool(line["option_text"])
        else:
            line["option_label"] = None
            line["option_text"] = ""
            line["label_only"] = False

    return sorted(lines, key=lambda line: (line["y"], line["x"]))


class LayoutPreflightService:
    """Analyze a cropped manual region before trusting provider reading order."""

    def analyze(
        self,
        *,
        region_id: str,
        text_items: List[Dict[str, Any]],
        embedded_images: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        lines = group_text_items_into_lines(text_items)
        has_text_layer = bool(lines)
        labels_found = [line["option_label"] for line in lines if line.get("option_label")]
        label_only_lines = [
            line["option_label"]
            for line in lines
            if line.get("option_label") and line.get("label_only")
        ]
        same_baseline_pairs = sum(
            1
            for line in lines
            if line.get("option_label") and not line.get("label_only")
        )
        unlabelled_option_lines = 0
        ambiguous_label_only = 0

        for idx, line in enumerate(lines):
            if line.get("option_label"):
                continue
            if not line.get("text"):
                continue
            next_line = lines[idx + 1] if idx + 1 < len(lines) else None
            if next_line and next_line.get("label_only"):
                unlabelled_option_lines += 1
                x_gap = abs(float(line.get("x", 0)) - float(next_line.get("x", 0)))
                y_gap = abs(float(next_line.get("y", 0)) - float(line.get("y", 0)))
                if x_gap > 40 or y_gap > 40:
                    ambiguous_label_only += 1

        unique_labels = sorted(set(labels_found))
        layout_risks: List[str] = []
        if (
            has_text_layer
            and len(unique_labels) >= 3
            and label_only_lines
            and unlabelled_option_lines
        ):
            layout_risks.append("staggered_options")
        if embedded_images:
            layout_risks.append("formula_or_image_dependency")

        confidence = 0.0
        if has_text_layer:
            confidence = min(
                0.95,
                0.35
                + (0.10 * len(unique_labels))
                + (0.12 * len(label_only_lines))
                + (0.08 * unlabelled_option_lines)
                - (0.10 * ambiguous_label_only),
            )

        recommended_strategy = "ocr_then_validate"
        if "staggered_options" in layout_risks:
            recommended_strategy = "ocr_with_layout_hints"

        return {
            "region_id": region_id,
            "has_text_layer": has_text_layer,
            "has_embedded_images": bool(embedded_images),
            "layout_risks": layout_risks,
            "option_layout": {
                "labels_found": unique_labels,
                "label_only_lines": label_only_lines,
                "unlabelled_option_lines": unlabelled_option_lines,
                "same_baseline_label_text_pairs": same_baseline_pairs,
                "confidence": round(max(0.0, min(1.0, confidence)), 2),
            },
            "visual_lines": [
                {
                    "text": line["text"],
                    "x": line["x"],
                    "y": line["y"],
                    "label": line.get("option_label"),
                    "label_only": line.get("label_only", False),
                }
                for line in lines
            ],
            "recommended_strategy": recommended_strategy,
        }
