from api.v1.practice_async import (
    _clean_model_solution_text,
    _decode_text_transport_escapes,
    _normalize_latex_for_render,
    _prepare_attempt_display_fields,
)


def test_clean_model_solution_preserves_latex_text_and_right():
    raw = (
        r'{"solution":"Compute $x_{\\text{CM}}$ and conclude '
        r'$\\left(x_{\\text{CM}}, y_{\\text{CM}}\\right)$."}'
    )

    cleaned = _clean_model_solution_text(raw)

    assert r"\text{CM}" in cleaned
    assert r"\right" in cleaned
    assert "\t" not in cleaned
    assert "ight)$" not in cleaned.replace(r"\right)$", "")


def test_normalize_repairs_json_control_damaged_latex():
    damaged = (
        "Thus $x_{"
        + "\t"
        + "ext{CM}}$ and $\\left(x_{"
        + "\t"
        + "ext{CM}}"
        + "\r"
        + "ight)$."
    )

    normalized = _normalize_latex_for_render(damaged)

    assert r"\text{CM}" in normalized
    assert r"\right" in normalized
    assert "\t" not in normalized
    assert "\r" not in normalized


def test_normalize_repairs_fraction_shorthand_from_model_text():
    normalized = _normalize_latex_for_render("The point is C(frac12, frac√32).")

    assert r"$\frac{1}{2}$" in normalized
    assert r"$\frac{\sqrt{3}}{2}$" in normalized


def test_decode_transport_turns_n_step_into_line_breaks_not_latex():
    raw = (
        r"To write about your state:\nStep 1: Understand the task."
        r"\nStep 2: Decide which state to describe."
        r"\nExample (for Karnataka):\nKarnataka lies in the southern region."
    )

    decoded = _decode_text_transport_escapes(raw)

    assert "\\nStep" not in decoded
    assert "\nStep 1:" in decoded
    assert "\nExample (for Karnataka):" in decoded
    assert "\nKarnataka lies" in decoded
    assert _decode_text_transport_escapes(r"x \neq y") == r"x \neq y"


def test_history_attempt_display_decodes_cached_solution_escapes():
    attempt = _prepare_attempt_display_fields({
        "correct_solution": r"Step 1: Outline.\nStep 2: Write.\nExample: Karnataka.",
        "what_went_wrong": r"Too short.\nAdd culture and features.",
        "score": 0.2,
    })

    assert "\\nStep" not in attempt["correct_solution"]
    assert "\nStep 2:" in attempt["correct_solution"]
    assert "\nAdd culture" in attempt["what_went_wrong"]
    assert attempt["score"] == 0.2

