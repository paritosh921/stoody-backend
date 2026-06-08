from api.v1.practice_async import _clean_model_solution_text, _normalize_latex_for_render


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

