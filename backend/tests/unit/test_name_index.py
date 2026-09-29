"""Unit tests for ``app.services.resolver.name_index``."""

from __future__ import annotations

import pytest

from app.services.resolver.name_index import NameIndex, NameMatch
from tests.fixtures.fake_repository import FakeCardRepository

ARMADES_ID = 88033975
DARK_MAGICIAN_ID = 46986414
DARK_MAGICIAN_GIRL_ID = 38033121
BLUE_EYES_ID = 89631139


@pytest.fixture()
def repo(ygoprodeck_sample: dict) -> FakeCardRepository:
    return FakeCardRepository.from_payload(ygoprodeck_sample)


@pytest.fixture()
def index(repo: FakeCardRepository) -> NameIndex:
    return NameIndex.from_repository(repo)


def test_index_size_matches_repository(index: NameIndex, repo: FakeCardRepository) -> None:
    assert len(index) == repo.count_cards() == 7


def test_exact_match_scores_one_and_ranks_first(index: NameIndex) -> None:
    matches = index.search("armades keeper of boundaries")
    assert matches[0] == NameMatch(card_id=ARMADES_ID, normalized_name="armades keeper of boundaries", score=1.0)
    assert matches[0].is_exact


def test_search_normalizes_query_defensively(index: NameIndex) -> None:
    matches = index.search("ARMADES, KEEPER OF BOUNDARIES")
    assert matches[0].card_id == ARMADES_ID
    assert matches[0].score == 1.0


def test_dark_magician_ranks_above_dark_magician_girl(index: NameIndex) -> None:
    matches = index.search("dark magician", limit=2)
    assert [m.card_id for m in matches] == [DARK_MAGICIAN_ID, DARK_MAGICIAN_GIRL_ID]
    assert matches[0].score == 1.0
    # A superset name must not be a perfect match (no partial_ratio).
    assert matches[1].score < 0.9


def test_query_of_longer_name_finds_it_exactly(index: NameIndex) -> None:
    matches = index.search("dark magician girl", limit=2)
    assert matches[0].card_id == DARK_MAGICIAN_GIRL_ID
    assert matches[0].score == 1.0
    assert matches[1].card_id == DARK_MAGICIAN_ID


def test_ocr_noise_still_finds_card(index: NameIndex) -> None:
    matches = index.search("armaoes keeper 0f b0undaries", limit=1)
    assert matches[0].card_id == ARMADES_ID
    assert matches[0].score >= 0.85


def test_token_order_insensitive(index: NameIndex) -> None:
    matches = index.search("white dragon blue eyes", limit=1)
    assert matches[0].card_id == BLUE_EYES_ID
    assert matches[0].score >= 0.9


def test_score_cutoff_filters(index: NameIndex) -> None:
    assert index.search("dark magician", score_cutoff=0.95) == [
        NameMatch(card_id=DARK_MAGICIAN_ID, normalized_name="dark magician", score=1.0)
    ]
    assert index.search("zzzz qqqq", score_cutoff=0.7) == []


def test_limit_respected_and_sorted_descending(index: NameIndex) -> None:
    matches = index.search("dark magician", limit=3, score_cutoff=0.0)
    assert len(matches) == 3
    assert [m.score for m in matches] == sorted((m.score for m in matches), reverse=True)


def test_empty_query_and_zero_limit(index: NameIndex) -> None:
    assert index.search("") == []
    assert index.search("   ") == []
    assert index.search("dark magician", limit=0) == []


def test_empty_index() -> None:
    assert len(NameIndex()) == 0
    assert NameIndex().search("dark magician") == []


def test_constructor_normalizes_entries_and_skips_blank() -> None:
    index = NameIndex([(1, "Dark Magician"), (2, "   "), (3, "Pot of Greed")])
    assert len(index) == 2
    assert index.search("dark magician")[0] == NameMatch(card_id=1, normalized_name="dark magician", score=1.0)


def test_tie_break_shorter_name_then_card_id() -> None:
    index = NameIndex([(20, "harpie lady 2"), (10, "harpie lady 1"), (30, "harpie lady sisters")])
    matches = index.search("harpie lady")
    assert [m.card_id for m in matches[:2]] == [10, 20]
    assert matches[0].score == pytest.approx(matches[1].score)
    assert matches[2].card_id == 30


def test_refresh_picks_up_new_cards(index: NameIndex, repo: FakeCardRepository) -> None:
    repo.add_card(900001, "Harpie Lady 1")
    assert index.search("harpie lady 1", score_cutoff=0.9) == []
    index.refresh(repo)
    assert len(index) == 8
    assert index.search("harpie lady 1", score_cutoff=0.9)[0].card_id == 900001


def test_scores_are_bounded(index: NameIndex) -> None:
    for match in index.search("blue eyes", score_cutoff=0.0):
        assert 0.0 <= match.score <= 1.0
