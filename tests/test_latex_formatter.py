from utils.latex_formatter import format_latex_in_text


def test_wraps_raw_coordinate_latex_with_dfrac_and_left_right():
    text = r"Option (B): \left(\dfrac{7}{12}\,\mathrm{m}, \dfrac{\sqrt{3}}{4}\,\mathrm{m}\right)"

    assert format_latex_in_text(text) == (
        r"Option (B): $\left(\dfrac{7}{12}\,\mathrm{m}, \dfrac{\sqrt{3}}{4}\,\mathrm{m}\right)$"
    )


def test_preserves_existing_delimited_math():
    text = r"Already $x_0$ and raw \sqrt{3}"

    assert format_latex_in_text(text) == r"Already $x_0$ and raw $\sqrt{3}$"


def test_wraps_chemistry_mhchem_notation():
    text = r"Correct formula: \ce{CO2 + H2O -> H2CO3}"

    assert format_latex_in_text(text) == r"Correct formula: $\ce{CO2 + H2O -> H2CO3}$"


def test_plain_subject_text_is_unchanged():
    text = "The answer is butan-2-ol because the OH group is on the second carbon."

    assert format_latex_in_text(text) == text
