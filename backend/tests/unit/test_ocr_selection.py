"""Reading-selection rules (set-code preference, name plausibility, alternatives)."""

from __future__ import annotations

import pytest

from app.services.ocr.base import OCRAlternative
from app.services.ocr.selection import name_plausibility, select_best_reading, set_code_plausibility


def test_empty_inputs_give_empty_result() -> None:
    assert select_best_reading([], "name").is_empty
    result = select_best_reading([("a", "", 0.9, None), ("b", "   ", 0.99, None)], "set_code")
    assert result.is_empty and result.confidence == 0.0 and result.alternatives == []


def test_unknown_kind_rejected() -> None:
    with pytest.raises(ValueError):
        select_best_reading([("a", "x", 0.5, None)], "other")  # type: ignore[arg-type]


# ------------------------------------------------------------ set code
def test_set_code_prefers_plausible_shape_over_confidence() -> None:
    readings = [
        ("raw", "1ST EDITION", 0.99, {"v": "raw"}),  # the strip next to the code; not a set-code shape
        ("up2", "JOTL-EN045", 0.80, {"v": "up2"}),
        ("otsu", "garbage", 0.95, {"v": "otsu"}),
    ]
    result = select_best_reading(readings, "set_code")
    assert result.text == "JOTL-EN045"
    assert result.confidence == 0.80
    assert result.variant == "up2"
    assert result.raw_result == {"v": "up2"}
    assert [a.text for a in result.alternatives] == ["1ST EDITION", "garbage"]
    assert all(isinstance(a, OCRAlternative) for a in result.alternatives)


def test_set_code_highest_confidence_among_plausible() -> None:
    readings = [("a", "JOTL-EN045", 0.91, None), ("b", "J0TL-EN045", 0.97, None), ("c", "JOTL-EN045", 0.95, None)]
    result = select_best_reading(readings, "set_code")
    assert result.text == "J0TL-EN045" and result.confidence == 0.97
    # duplicates collapse to the best-scoring copy
    assert [(a.text, a.confidence) for a in result.alternatives] == [("JOTL-EN045", 0.95)]


def test_set_code_hyphenless_counts_as_plausible() -> None:
    assert set_code_plausibility("JOTLEN045") == 1.0
    assert set_code_plausibility("JOTL-EN045") == 1.0
    assert set_code_plausibility("hello world") == 0.0
    readings = [("a", "JOTLEN045", 0.6, None), ("b", "xx", 0.99, None)]
    assert select_best_reading(readings, "set_code").text == "JOTLEN045"


def test_set_code_falls_back_to_confidence_without_plausible_reading() -> None:
    readings = [("a", "??", 0.4, None), ("b", "!!", 0.7, None)]
    result = select_best_reading(readings, "set_code")
    assert result.text == "!!" and result.variant == "b"
    assert [a.text for a in result.alternatives] == ["??"]


# ------------------------------------------------------------ name
def test_name_plausibility_penalties() -> None:
    assert name_plausibility("") == 0.0
    assert name_plausibility("--- ,,,") == pytest.approx(0.05)
    assert name_plausibility("A1") == pytest.approx(0.30)  # < 3 letters
    full = name_plausibility("Armades, Keeper of Boundaries")
    assert 0.95 <= full <= 1.0
    assert name_plausibility("ABC1234567") < 0.5  # letters+spaces < 60 %
    assert name_plausibility("Kuriboh") < full  # mild length bonus
    assert name_plausibility("Kuriboh") > 0.85


def test_name_selection_prefers_plausible_text() -> None:
    readings = [
        ("raw", "--", 0.99, None),
        ("tight", "A1", 0.99, None),
        ("inv", "Kuriboh", 0.80, {"v": "inv"}),
    ]
    result = select_best_reading(readings, "name")
    assert result.text == "Kuriboh" and result.variant == "inv" and result.raw_result == {"v": "inv"}
    assert {a.text for a in result.alternatives} == {"--", "A1"}


def test_name_selection_uses_confidence_among_plausible_and_keeps_alternatives() -> None:
    readings = [
        ("a", "ARMADES, KEEPER OF BOUNDARIES", 0.90, None),
        ("b", "ARMADES. KEEPER OF BOUNDARIES", 0.93, None),
        ("c", "ARMADES, KEEPER OF BOUNDARIES", 0.95, None),
    ]
    result = select_best_reading(readings, "name")
    assert result.text == "ARMADES, KEEPER OF BOUNDARIES" and result.confidence == 0.95 and result.variant == "c"
    assert [(a.text, a.variant) for a in result.alternatives] == [("ARMADES. KEEPER OF BOUNDARIES", "b")]
    assert result.all_readings()[0] == ("ARMADES, KEEPER OF BOUNDARIES", 0.95)


def test_name_selection_strips_whitespace() -> None:
    result = select_best_reading([("a", "  Kuriboh \n", 0.5, None)], "name")
    assert result.text == "Kuriboh" and result.alternatives == []
