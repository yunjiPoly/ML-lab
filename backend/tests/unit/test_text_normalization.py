"""Unit tests for ``app.core.text`` (shared normalization contract)."""

from __future__ import annotations

import pytest

from app.core.text import (
    SET_CODE_RE,
    looks_like_set_code,
    normalize_card_name,
    normalize_set_code,
    set_code_prefix,
    set_code_variants,
    split_set_code,
    strip_accents,
)

# --------------------------------------------------------------- card names


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Armades, Keeper of Boundaries", "armades keeper of boundaries"),
        ("ARMADES KEEPER OF BOUNDARIES", "armades keeper of boundaries"),
        ("Blue-Eyes White Dragon", "blue eyes white dragon"),
        ("D.D. Warrior Lady", "d d warrior lady"),
        ("Number 39: Utopia", "number 39 utopia"),
        ("  Pot   of\tGreed \n", "pot of greed"),
        ("Rock & Roll", "rock and roll"),
        ("Pokémon-Élan", "pokemon elan"),
        ("Ｄａｒｋ Ｍａｇｉｃｉａｎ", "dark magician"),  # full-width letters
        ("Harpie's Pet Dragon", "harpie s pet dragon"),
        ("!!!", ""),
    ],
)
def test_normalize_card_name(raw: str, expected: str) -> None:
    assert normalize_card_name(raw) == expected


def test_normalize_card_name_handles_none_and_empty() -> None:
    assert normalize_card_name(None) == ""
    assert normalize_card_name("") == ""


def test_normalize_card_name_is_idempotent() -> None:
    once = normalize_card_name("Armades, Keeper of Boundaries!")
    assert normalize_card_name(once) == once


def test_strip_accents() -> None:
    assert strip_accents("Éléphant à Noël") == "Elephant a Noel"


# ---------------------------------------------------------------- set codes


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("JOTL-EN045", "JOTL-EN045"),
        ("jotl-en045", "JOTL-EN045"),
        (" JOTL - EN045 ", "JOTL-EN045"),
        ("JOTL–EN045", "JOTL-EN045"),  # en dash
        ("JOTL—EN045", "JOTL-EN045"),  # em dash
        ("JOTL−EN045", "JOTL-EN045"),  # unicode minus
        ("JOTL_EN045", "JOTL-EN045"),  # underscore
        ("JOTL--EN045", "JOTL-EN045"),
        ("-JOTL-EN045-", "JOTL-EN045"),
        ("JOTL-EN045.", "JOTL-EN045"),
        ("(JOTL-EN045)", "JOTL-EN045"),
        ("ＪＯＴＬ－ＥＮ０４５", "JOTL-EN045"),  # full-width
        ("LOB-001", "LOB-001"),
        ("duea-ense1", "DUEA-ENSE1"),
        ("---", ""),
    ],
)
def test_normalize_set_code(raw: str, expected: str) -> None:
    assert normalize_set_code(raw) == expected


def test_normalize_set_code_handles_none_and_empty() -> None:
    assert normalize_set_code(None) == ""
    assert normalize_set_code("") == ""


def test_normalize_set_code_does_not_fix_ocr_confusions() -> None:
    assert normalize_set_code("J0TL-ENO45") == "J0TL-ENO45"


@pytest.mark.parametrize(
    "code",
    ["JOTL-EN045", "LOB-001", "DUEA-ENSE1", "MP14-EN095", "2020-EN001", "SDK-E001", "LDK2-ENK01", "jotl-en045", "MVP1-ENSV4"],
)
def test_looks_like_set_code_accepts_known_shapes(code: str) -> None:
    assert looks_like_set_code(code)
    assert SET_CODE_RE.match(normalize_set_code(code))


@pytest.mark.parametrize("code", ["", None, "hello", "JOTLEN045", "J-EN045", "JOTL-EN", "ABCDEF-EN045", "JOTL-EN045678"])
def test_looks_like_set_code_rejects_other_shapes(code: str | None) -> None:
    assert not looks_like_set_code(code)


def test_split_set_code_and_prefix() -> None:
    assert split_set_code("JOTL-EN045") == ("JOTL", "EN045")
    assert split_set_code("LOB-001") == ("LOB", "001")
    assert split_set_code("nope") is None
    assert set_code_prefix("JOTL-EN045") == "JOTL"
    assert set_code_prefix("jotl-en045") == "JOTL"
    assert set_code_prefix("garbage") == "GARBAGE"


# ------------------------------------------------------------------ variants


def test_variants_first_entry_is_as_is() -> None:
    variants = set_code_variants("J0TL-ENO45")
    assert variants[0] == "J0TL-ENO45"
    assert "JOTL-EN045" in variants


def test_variants_clean_code_still_first_and_unique() -> None:
    variants = set_code_variants("JOTL-EN045")
    assert variants[0] == "JOTL-EN045"
    assert len(variants) == len(set(variants))
    assert all(SET_CODE_RE.match(v) for v in variants)


def test_variants_reinsert_dropped_hyphen_modern() -> None:
    variants = set_code_variants("JOTLEN045")
    assert variants[0] == "JOTL-EN045"


def test_variants_reinsert_dropped_hyphen_with_confusions() -> None:
    assert "JOTL-EN045" in set_code_variants("JOTLENO45")


def test_variants_reinsert_dropped_hyphen_legacy() -> None:
    assert set_code_variants("LOB001")[0] == "LOB-001"
    assert set_code_variants("MRD071")[0] == "MRD-071"


def test_variants_legacy_code_kept_as_is() -> None:
    variants = set_code_variants("LOB-001")
    assert variants[0] == "LOB-001"
    # Suffix without region letters is never coerced.
    assert all(v.endswith("-001") for v in variants)


def test_variants_special_edition_suffix_kept_as_is() -> None:
    variants = set_code_variants("DUEA-ENSE1")
    assert variants[0] == "DUEA-ENSE1"


def test_variants_suffix_coercion_maps_letters_to_digits_and_back() -> None:
    assert "JOTL-EN045" in set_code_variants("JOTL-ENO4S")  # number part: O->0, S->5
    assert "MP14-EN095" in set_code_variants("MP14-EN09S")
    assert "JOTL-EN045" not in set_code_variants("JOTL-3N045")  # 3 has no letter alternate: stays conservative
    assert "BLGG-EN090" in set_code_variants("8LGG-EN090")  # prefix: 8->B


def test_variants_region_digit_confusions_are_mapped_back_to_letters() -> None:
    assert "MRD-EN071" not in set_code_variants("MRD-3N071")
    assert "SDK-EN001" in set_code_variants("5DK-EN001")  # prefix S read as 5
    assert "LOB-EN001" in set_code_variants("L0B-EN001")  # prefix 0 read for O


def test_variants_prefix_alternates_fewer_substitutions_first() -> None:
    variants = set_code_variants("JOTL-EN045")
    # One-substitution prefixes come before the two-substitution one.
    assert variants.index("JOT1-EN045") < variants.index("J0T1-EN045")
    assert variants.index("J0TL-EN045") < variants.index("J0T1-EN045")


def test_variants_extract_code_token_from_longer_reading() -> None:
    assert set_code_variants("1st Edition JOTL-EN045")[0] == "JOTL-EN045"
    assert set_code_variants("JOTL-EN045 ScR")[0] == "JOTL-EN045"


def test_variants_do_not_extract_tokens_when_whole_reading_is_a_code() -> None:
    variants = set_code_variants("JOTL - EN045")
    assert variants[0] == "JOTL-EN045"
    assert "EN-045" not in variants


@pytest.mark.parametrize("raw", ["", None, "hello", "!!!", "12"])
def test_variants_empty_for_unusable_input(raw: str | None) -> None:
    assert set_code_variants(raw) == []


def test_variants_respect_limit() -> None:
    assert len(set_code_variants("BLGG-EN090", limit=3)) == 3
    assert set_code_variants("BLGG-EN090", limit=3)[0] == "BLGG-EN090"


def test_variants_unicode_dash_and_case() -> None:
    assert set_code_variants("jotl–en045")[0] == "JOTL-EN045"


class TestNormalizeRarityCode:
    def test_strips_parentheses_and_whitespace(self) -> None:
        from app.core.text import normalize_rarity_code

        assert normalize_rarity_code("(ScR)") == "ScR"
        assert normalize_rarity_code(" (UR) ") == "UR"
        assert normalize_rarity_code("C") == "C"

    def test_empty_values_become_none(self) -> None:
        from app.core.text import normalize_rarity_code

        assert normalize_rarity_code(None) is None
        assert normalize_rarity_code("") is None
        assert normalize_rarity_code("()") is None
