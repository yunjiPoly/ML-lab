"""Unit tests for ``app.services.resolver.resolver`` (every decision branch)."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.schemas.recognition import RecognitionStatus, ResolutionResult
from app.services.ocr.base import OCRAlternative, OCRResult
from app.services.resolver import NameIndex, Resolver, name_similarity
from app.services.visual.base import VisualCandidate
from tests.fixtures.fake_repository import FakeCardRepository

ARMADES = "Armades, Keeper of Boundaries"
ARMADES_ID = 88033975
DARK_MAGICIAN_ID = 46986414
KURIBOH_ID = 40640057
HARPIE_1_ID = 900001
HARPIE_2_ID = 900002
LONELY_ID = 900003  # synthetic card with exactly one printing


@pytest.fixture()
def repo(ygoprodeck_sample: dict) -> FakeCardRepository:
    repo = FakeCardRepository.from_payload(ygoprodeck_sample)
    repo.add_card(HARPIE_1_ID, "Harpie Lady 1", printings=[("LOB-EN049", "Legend of Blue Eyes White Dragon", "Common", "C")])
    repo.add_card(HARPIE_2_ID, "Harpie Lady 2", printings=[("MRD-EN010", "Metal Raiders", "Common", "C")])
    repo.add_card(LONELY_ID, "Lonely Test Printing", printings=[("TEST-EN001", "Test Set", "Rare", "R")])
    return repo


@pytest.fixture()
def resolver(repo: FakeCardRepository, test_settings: Settings) -> Resolver:
    return Resolver(repo, test_settings)


def ocr(text: str, confidence: float, **kwargs) -> OCRResult:
    return OCRResult(text=text, confidence=confidence, **kwargs)


def assert_well_formed(result: ResolutionResult) -> None:
    assert 0.0 <= result.confidence <= 1.0
    for candidate in result.candidates:
        assert 0.0 <= candidate.score <= 1.0
        assert candidate.reasons


# ------------------------------------------------------------ set-code branch


def test_canonical_example_is_matched(resolver: Resolver) -> None:
    result = resolver.resolve(ocr("ARMADES KEEPER OF BOUNDARIES", 0.70), ocr("J0TL-ENO45", 0.95))
    assert_well_formed(result)
    assert result.status == RecognitionStatus.MATCHED
    assert result.card is not None and result.card.name == ARMADES and result.card.id == ARMADES_ID
    assert result.printing is not None
    assert (result.printing.set_code, result.printing.set_name, result.printing.rarity) == (
        "JOTL-EN045", "Judgment of the Light", "Secret Rare",
    )
    assert result.set_code is not None
    assert result.set_code.raw == "J0TL-ENO45"
    assert result.set_code.normalized == "JOTL-EN045"
    assert "J0TL-ENO45" in result.set_code.alternatives  # the as-is form was tried too
    assert result.name is not None and result.name.normalized == "armades keeper of boundaries"
    assert result.confidence >= 0.8
    assert result.candidates[0].printing is not None and result.candidates[0].printing.set_code == "JOTL-EN045"
    assert any("set_code_variant=JOTL-EN045" in r for r in result.candidates[0].reasons)
    assert any("corrected" in note for note in result.notes)


def test_exact_code_and_name_scores_higher_than_corrected_code(resolver: Resolver) -> None:
    exact = resolver.resolve(ocr(ARMADES, 0.70), ocr("JOTL-EN045", 0.95))
    corrected = resolver.resolve(ocr(ARMADES, 0.70), ocr("J0TL-ENO45", 0.95))
    assert exact.status == corrected.status == RecognitionStatus.MATCHED
    assert exact.confidence > corrected.confidence
    assert "set_code_exact" in exact.candidates[0].reasons
    assert not any("corrected" in note for note in exact.notes)


def test_two_rarities_share_set_code(resolver: Resolver) -> None:
    result = resolver.resolve(ocr(ARMADES, 0.9), ocr("BLGG-EN090", 0.9))
    assert_well_formed(result)
    assert result.status == RecognitionStatus.MATCHED
    assert result.printing is not None
    assert result.printing.set_code == "BLGG-EN090"
    assert result.printing.rarity == "Secret Rare"  # first rarity kept, never None
    rarities = {c.printing.rarity for c in result.candidates if c.printing and c.printing.set_code == "BLGG-EN090"}
    assert rarities == {"Secret Rare", "Starlight Rare"}
    assert any("2 rarities share this set code" in note for note in result.notes)
    assert "rarity_alternative" in result.candidates[1].reasons


def test_garbled_name_with_exact_confident_code_trusts_set_code(resolver: Resolver) -> None:
    result = resolver.resolve(ocr("XQZ PLUGH WIBBLE", 0.9), ocr("JOTL-EN045", 0.95))
    assert_well_formed(result)
    assert result.status == RecognitionStatus.MATCHED
    assert result.card is not None and result.card.id == ARMADES_ID
    assert result.printing is not None and result.printing.set_code == "JOTL-EN045"
    assert any("name OCR disagreed; trusting printed set code" in note for note in result.notes)
    agreeing = resolver.resolve(ocr(ARMADES, 0.9), ocr("JOTL-EN045", 0.95))
    assert result.confidence < agreeing.confidence


def test_garbled_name_with_exact_but_low_confidence_code_is_low_confidence(resolver: Resolver) -> None:
    result = resolver.resolve(ocr("XQZ PLUGH WIBBLE", 0.9), ocr("JOTL-EN045", 0.5))
    assert result.status == RecognitionStatus.LOW_CONFIDENCE
    assert result.card is not None and result.card.id == ARMADES_ID


def test_garbled_name_with_corrected_code_is_low_confidence(resolver: Resolver) -> None:
    result = resolver.resolve(ocr("XQZ PLUGH WIBBLE", 0.9), ocr("J0TL-ENO45", 0.95))
    assert_well_formed(result)
    assert result.status == RecognitionStatus.LOW_CONFIDENCE
    assert result.card is not None and result.card.id == ARMADES_ID  # primary suggestion
    assert result.printing is not None and result.printing.set_code == "JOTL-EN045"
    assert any("verify manually" in note for note in result.notes)


def test_hyphen_dropped_code_is_matched(resolver: Resolver) -> None:
    result = resolver.resolve(ocr(ARMADES, 0.9), ocr("JOTLEN045", 0.9))
    assert result.status == RecognitionStatus.MATCHED
    assert result.printing is not None and result.printing.set_code == "JOTL-EN045"
    assert result.set_code is not None and result.set_code.raw == "JOTLEN045"


def test_unusable_name_with_exact_confident_code_is_matched(resolver: Resolver) -> None:
    result = resolver.resolve(ocr(ARMADES, 0.10), ocr("JOTL-EN045", 0.95))
    assert result.status == RecognitionStatus.MATCHED
    assert result.card is not None and result.card.id == ARMADES_ID
    assert any("below minimum" in note for note in result.notes)
    assert any("name OCR unusable; trusting printed set code" in note for note in result.notes)


def test_no_name_with_corrected_code_is_low_confidence(resolver: Resolver) -> None:
    result = resolver.resolve(None, ocr("J0TL-ENO45", 0.95))
    assert result.status == RecognitionStatus.LOW_CONFIDENCE
    assert result.name is None
    assert result.card is not None and result.card.id == ARMADES_ID


def test_low_confidence_code_that_looks_like_a_code_is_still_used(resolver: Resolver) -> None:
    result = resolver.resolve(ocr(ARMADES, 0.9), ocr("JOTL-EN045", 0.2))
    assert result.status == RecognitionStatus.MATCHED
    assert result.printing is not None and result.printing.set_code == "JOTL-EN045"


def test_alternative_set_code_reading_is_used(resolver: Resolver) -> None:
    code = ocr("JQTL-EN0A5", 0.5, variant="raw", alternatives=[OCRAlternative("JOTL-EN045", 0.45, "binarized")])
    result = resolver.resolve(ocr(ARMADES, 0.9), code)
    assert result.status == RecognitionStatus.MATCHED
    assert result.printing is not None and result.printing.set_code == "JOTL-EN045"
    assert result.set_code is not None and result.set_code.raw == "JQTL-EN0A5"
    assert result.set_code.normalized == "JOTL-EN045"
    assert any("alternative OCR reading" in note for note in result.notes)


def test_set_code_hits_on_two_cards_name_decides(repo: FakeCardRepository, test_settings: Settings) -> None:
    # A second card whose code is a confusion-variant of Armades' JOTL-EN045.
    repo.add_card(900010, "Decoy Dragon", printings=[("J0TL-EN045", "Fake Set", "Common", "C")])
    resolver = Resolver(repo, test_settings)
    result = resolver.resolve(ocr(ARMADES, 0.9), ocr("J0TL-EN045", 0.9))
    assert result.status == RecognitionStatus.MATCHED
    assert result.card is not None and result.card.id == ARMADES_ID
    assert any("matched 2 cards" in note for note in result.notes)


def test_set_code_hits_on_two_cards_without_name_is_ambiguous(repo: FakeCardRepository, test_settings: Settings) -> None:
    repo.add_card(900010, "Decoy Dragon", printings=[("J0TL-EN045", "Fake Set", "Common", "C")])
    resolver = Resolver(repo, test_settings)
    result = resolver.resolve(ocr("XQZ PLUGH", 0.9), ocr("JOTL-EN045", 0.9))
    assert_well_formed(result)
    assert result.status == RecognitionStatus.AMBIGUOUS
    assert result.card is None
    assert {c.card.id for c in result.candidates} == {ARMADES_ID, 900010}
    assert result.candidates[0].card.id == ARMADES_ID  # the exact as-is hit ranks first


# ---------------------------------------------------------------- name branch


def test_strong_name_with_unreadable_code_is_low_confidence_with_printings(resolver: Resolver, test_settings: Settings) -> None:
    result = resolver.resolve(ocr("ARMADES KEEPER OF BOUNDARIES", 0.9), ocr("", 0.0))
    assert_well_formed(result)
    assert result.status == RecognitionStatus.LOW_CONFIDENCE
    assert result.card is not None and result.card.id == ARMADES_ID
    assert result.printing is None
    assert len(result.candidates) == test_settings.resolver_max_candidates
    assert all(c.card.id == ARMADES_ID and c.printing is not None for c in result.candidates)
    assert all("printing_unverified" in c.reasons for c in result.candidates)
    assert any("printing undetermined" in note for note in result.notes)
    assert result.confidence == pytest.approx(0.9 * 0.8)


def test_strong_name_with_no_code_field_at_all(resolver: Resolver) -> None:
    result = resolver.resolve(ocr(ARMADES, 0.9), None)
    assert result.status == RecognitionStatus.LOW_CONFIDENCE
    assert result.set_code is None
    assert result.card is not None and result.card.id == ARMADES_ID


def test_strong_name_with_fuzzy_code_is_matched(resolver: Resolver) -> None:
    result = resolver.resolve(ocr(ARMADES, 0.9), ocr("JOTL-EN046", 0.8))  # no such printing; one digit off
    assert_well_formed(result)
    assert result.status == RecognitionStatus.MATCHED
    assert result.printing is not None and result.printing.set_code == "JOTL-EN045"
    assert result.set_code is not None and result.set_code.normalized == "JOTL-EN045"
    assert any("matched fuzzily" in note for note in result.notes)
    exact = resolver.resolve(ocr(ARMADES, 0.9), ocr("JOTL-EN045", 0.8))
    assert result.confidence < exact.confidence


def test_strong_name_with_far_off_code_lists_closest_printings_first(resolver: Resolver) -> None:
    result = resolver.resolve(ocr(ARMADES, 0.9), ocr("XXXX-EN045", 0.8))  # similarity 0.6 < 0.75
    assert result.status == RecognitionStatus.LOW_CONFIDENCE
    assert result.printing is None
    assert result.candidates[0].printing is not None and result.candidates[0].printing.set_code == "JOTL-EN045"
    assert any(r.startswith("set_code_similarity=") for r in result.candidates[0].reasons)


def test_single_printing_card_assumes_its_printing(resolver: Resolver) -> None:
    result = resolver.resolve(ocr("LONELY TEST PRINTING", 0.9), ocr("", 0.0))
    assert result.status == RecognitionStatus.LOW_CONFIDENCE
    assert result.card is not None and result.card.id == LONELY_ID
    assert result.printing is not None and result.printing.set_code == "TEST-EN001"
    assert any("assumed" in note for note in result.notes)


def test_dark_magician_exact_beats_dark_magician_girl(resolver: Resolver) -> None:
    result = resolver.resolve(ocr("DARK MAGICIAN", 0.9), None)
    assert result.status != RecognitionStatus.AMBIGUOUS
    assert result.status == RecognitionStatus.LOW_CONFIDENCE  # card known, printing not
    assert result.card is not None and result.card.id == DARK_MAGICIAN_ID
    assert all(c.card.id == DARK_MAGICIAN_ID for c in result.candidates)


def test_truly_ambiguous_names(resolver: Resolver) -> None:
    result = resolver.resolve(ocr("HARPIE LADY", 0.9), None)  # digit lost: "Harpie Lady 1" vs "Harpie Lady 2"
    assert_well_formed(result)
    assert result.status == RecognitionStatus.AMBIGUOUS
    assert result.card is None and result.printing is None
    assert {c.card.id for c in result.candidates} == {HARPIE_1_ID, HARPIE_2_ID}
    assert any("within" in note for note in result.notes)


def test_name_between_thresholds_is_low_confidence_with_candidates(resolver: Resolver, test_settings: Settings) -> None:
    query = "ARMADES KEEPER OF"  # truncated reading: similarity ~0.76
    score = name_similarity(query, "armades keeper of boundaries")
    assert test_settings.resolver_name_candidate_threshold <= score < test_settings.resolver_name_strong_threshold
    result = resolver.resolve(ocr(query, 0.9), None)
    assert result.status == RecognitionStatus.LOW_CONFIDENCE
    assert result.card is None
    assert result.candidates and result.candidates[0].card.id == ARMADES_ID
    assert any("below the strong threshold" in note for note in result.notes)


def test_alternative_name_reading_is_used(resolver: Resolver) -> None:
    name = ocr("ZQX WIBBLE", 0.6, alternatives=[OCRAlternative("Armades Keeper of Boundaries", 0.55, "sharpened")])
    result = resolver.resolve(name, None)
    assert result.status == RecognitionStatus.LOW_CONFIDENCE
    assert result.card is not None and result.card.id == ARMADES_ID
    assert result.name is not None and "armades keeper of boundaries" in result.name.alternatives
    assert any("alternative OCR reading" in note for note in result.notes)


# ------------------------------------------------------------- nothing branch


def test_nonsense_is_not_found(resolver: Resolver) -> None:
    result = resolver.resolve(ocr("ZZZZ QQQQ", 0.9), ocr("ZZZZ-QQ999", 0.9))
    assert result.status == RecognitionStatus.NOT_FOUND
    assert result.card is None and result.printing is None
    assert result.confidence == 0.0
    assert result.candidates == []
    assert any("no card matched" in note for note in result.notes)


def test_empty_ocr_is_ocr_failed(resolver: Resolver) -> None:
    result = resolver.resolve(ocr("", 0.0), ocr("", 0.0))
    assert result.status == RecognitionStatus.OCR_FAILED
    assert result.confidence == 0.0
    assert result.name is not None and result.name.raw == ""
    assert result.set_code is not None and result.set_code.raw == ""


def test_missing_fields_are_ocr_failed(resolver: Resolver) -> None:
    result = resolver.resolve(None, None)
    assert result.status == RecognitionStatus.OCR_FAILED
    assert result.name is None and result.set_code is None


def test_both_fields_below_min_confidence_is_ocr_failed(resolver: Resolver) -> None:
    result = resolver.resolve(ocr(ARMADES, 0.1), ocr("??", 0.1))
    assert result.status == RecognitionStatus.OCR_FAILED
    assert len([n for n in result.notes if "below minimum" in n]) == 2


def test_visual_candidates_used_when_text_fails(resolver: Resolver) -> None:
    visual = [VisualCandidate(card_id=KURIBOH_ID, score=0.8), VisualCandidate(card_id=1, score=0.95)]  # id 1 unknown
    result = resolver.resolve(ocr("ZZZZ QQQQ", 0.9), None, visual_candidates=visual)
    assert result.status == RecognitionStatus.LOW_CONFIDENCE
    assert result.card is None
    assert [c.card.id for c in result.candidates] == [KURIBOH_ID]
    assert result.candidates[0].reasons == ["visual_similarity"]
    assert any("visual similarity" in note for note in result.notes)


def test_visual_candidates_ignored_when_text_matches(resolver: Resolver) -> None:
    visual = [VisualCandidate(card_id=KURIBOH_ID, score=0.99)]
    result = resolver.resolve(ocr(ARMADES, 0.9), ocr("JOTL-EN045", 0.9), visual_candidates=visual)
    assert result.status == RecognitionStatus.MATCHED
    assert result.card is not None and result.card.id == ARMADES_ID


# -------------------------------------------------------------- configuration


def test_candidates_capped_by_settings(repo: FakeCardRepository, test_settings: Settings) -> None:
    settings = test_settings.model_copy(update={"resolver_max_candidates": 2})
    result = Resolver(repo, settings).resolve(ocr("DARK MAGICIAN", 0.9), None)
    assert len(result.candidates) == 2


def test_injected_name_index_is_used(repo: FakeCardRepository, test_settings: Settings) -> None:
    index = NameIndex([(KURIBOH_ID, "kuriboh")])
    resolver = Resolver(repo, test_settings, name_index=index)
    assert resolver.name_index is index
    assert resolver.resolve(ocr(ARMADES, 0.9), None).status == RecognitionStatus.NOT_FOUND
    assert resolver.resolve(ocr("KURIBOH", 0.9), None).card is not None
