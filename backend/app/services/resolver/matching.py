"""Similarity primitives shared by the name index and the resolver.

All scores are floats in ``[0, 1]``.  They are *application-level* similarity
measures (rapidfuzz ratios), not calibrated probabilities.

* :func:`name_similarity` scores two normalized card names with the maximum of
  ``fuzz.ratio`` and ``fuzz.token_sort_ratio``.  ``partial_ratio`` is deliberately
  NOT used: it makes ``"dark magician"`` match ``"dark magician girl"`` at 1.0.
* :func:`set_code_similarity` is a normalized Levenshtein similarity on
  normalized set codes; it is used to *rank* the printings of an already
  identified card when the OCR'd code did not exist verbatim.
"""

from __future__ import annotations

from collections.abc import Sequence

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from app.core.text import normalize_card_name, normalize_set_code


def name_similarity(a: str, b: str) -> float:
    """Similarity of two card names in ``[0, 1]``.

    Inputs are normalized with :func:`normalize_card_name` (idempotent, so
    passing already-normalized strings is fine).  Identical normalized names
    score exactly ``1.0``; an empty side scores ``0.0``.
    """
    left = normalize_card_name(a)
    right = normalize_card_name(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    score = max(fuzz.ratio(left, right), fuzz.token_sort_ratio(left, right))
    return min(1.0, max(0.0, score / 100.0))


def set_code_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity of two set codes in ``[0, 1]``.

    Both sides are normalized with :func:`normalize_set_code` first.  Identical
    codes score ``1.0``; an empty side scores ``0.0``.
    """
    left = normalize_set_code(a)
    right = normalize_set_code(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return float(Levenshtein.normalized_similarity(left, right))


def closest_set_code(query: str, codes: Sequence[str]) -> tuple[str | None, float]:
    """Return ``(code, similarity)`` for the entry of ``codes`` closest to ``query``.

    Ties keep the first occurrence so callers can pass codes in preference
    order.  Returns ``(None, 0.0)`` when ``query`` or ``codes`` is empty.  The
    returned code is the *original* entry of ``codes`` (not re-normalized).
    """
    if not codes or not normalize_set_code(query):
        return None, 0.0
    best_code: str | None = None
    best_score = -1.0
    for code in codes:
        score = set_code_similarity(query, code)
        if score > best_score:
            best_code, best_score = code, score
    return best_code, max(0.0, best_score)
