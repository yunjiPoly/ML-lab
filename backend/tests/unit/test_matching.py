"""Unit tests for ``app.services.resolver.matching``."""

from __future__ import annotations

import pytest

from app.services.resolver.matching import closest_set_code, name_similarity, set_code_similarity

# ---------------------------------------------------------- name_similarity


def test_name_similarity_identical_is_one() -> None:
    assert name_similarity("dark magician", "dark magician") == 1.0


def test_name_similarity_normalizes_inputs() -> None:
    assert name_similarity("ARMADES, KEEPER OF BOUNDARIES!", "armades keeper of boundaries") == 1.0


def test_name_similarity_empty_is_zero() -> None:
    assert name_similarity("", "dark magician") == 0.0
    assert name_similarity("dark magician", "") == 0.0
    assert name_similarity("", "") == 0.0


def test_name_similarity_superset_name_is_not_perfect() -> None:
    """partial_ratio would give 1.0 here; we must not."""
    score = name_similarity("dark magician", "dark magician girl")
    assert 0.75 < score < 0.90


def test_name_similarity_is_symmetric_and_bounded() -> None:
    a, b = "blue eyes white dragon", "red eyes black dragon"
    assert name_similarity(a, b) == pytest.approx(name_similarity(b, a))
    assert 0.0 <= name_similarity(a, b) <= 1.0


def test_name_similarity_token_order_insensitive() -> None:
    assert name_similarity("magician dark", "dark magician") >= 0.9


def test_name_similarity_tolerates_ocr_noise() -> None:
    assert name_similarity("armaoes keeper 0f b0undaries", "armades keeper of boundaries") >= 0.85


def test_name_similarity_unrelated_is_low() -> None:
    assert name_similarity("pot of greed", "mirror force") < 0.5


# ------------------------------------------------------ set_code_similarity


def test_set_code_similarity_identical() -> None:
    assert set_code_similarity("JOTL-EN045", "JOTL-EN045") == 1.0


def test_set_code_similarity_normalizes() -> None:
    assert set_code_similarity("jotl–en045", "JOTL-EN045") == 1.0


def test_set_code_similarity_single_edit() -> None:
    assert set_code_similarity("JOTL-EN046", "JOTL-EN045") == pytest.approx(0.9)
    assert set_code_similarity("J0TL-ENO45", "JOTL-EN045") == pytest.approx(0.8)


def test_set_code_similarity_empty_is_zero() -> None:
    assert set_code_similarity("", "JOTL-EN045") == 0.0
    assert set_code_similarity("JOTL-EN045", "") == 0.0


def test_set_code_similarity_unrelated_is_low() -> None:
    assert set_code_similarity("JOTL-EN045", "MP14-EN095") <= 0.5


# ---------------------------------------------------------- closest_set_code


def test_closest_set_code_picks_best() -> None:
    codes = ["MP14-EN095", "BLGG-EN090", "JOTL-EN045", "PGL2-EN043"]
    code, score = closest_set_code("JOTL-EN046", codes)
    assert code == "JOTL-EN045"
    assert score == pytest.approx(0.9)


def test_closest_set_code_exact_hit() -> None:
    code, score = closest_set_code("BLGG-EN090", ["JOTL-EN045", "BLGG-EN090"])
    assert (code, score) == ("BLGG-EN090", 1.0)


def test_closest_set_code_tie_keeps_first() -> None:
    code, score = closest_set_code("JOTL-EN045", ["JOTL-EN046", "JOTL-EN044"])
    assert code == "JOTL-EN046"
    assert score == pytest.approx(0.9)


def test_closest_set_code_returns_original_entry() -> None:
    code, _ = closest_set_code("JOTL-EN045", ["jotl-en045"])
    assert code == "jotl-en045"


def test_closest_set_code_empty_inputs() -> None:
    assert closest_set_code("JOTL-EN045", []) == (None, 0.0)
    assert closest_set_code("", ["JOTL-EN045"]) == (None, 0.0)
    assert closest_set_code("---", ["JOTL-EN045"]) == (None, 0.0)
