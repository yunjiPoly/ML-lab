"""Pick the best reading among several preprocessing variants.

The provider runs every variant of a crop through the recognizer and hands the
``(variant, text, confidence, raw)`` tuples to :func:`select_best_reading`,
which applies field-specific plausibility rules and keeps the rest as
alternatives so the resolver can still fall back on them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from app.core.text import looks_like_set_code, set_code_variants
from app.services.ocr.base import OCRAlternative, OCRResult

ReadingKind = Literal["name", "set_code"]
Reading = tuple[str, str, float, Any]
"""``(variant_name, text, confidence, raw_engine_result)``."""

_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
MIN_NAME_LETTERS = 3
MIN_NAME_LETTER_RATIO = 0.60


def name_plausibility(text: str) -> float:
    """Heuristic 0..1 score for how much ``text`` looks like a card name.

    * all punctuation / no alphanumerics -> 0.05
    * fewer than :data:`MIN_NAME_LETTERS` letters -> 0.30
    * less than :data:`MIN_NAME_LETTER_RATIO` letters+spaces -> scaled down
    * longer readings get a mild bonus (up to 15 %, saturating at 20 letters):
      CTC decoders can report decent confidence for truncated outputs, so this
      breaks near-ties in favour of the fuller reading without overriding a
      clear confidence gap.
    """
    stripped = text.strip()
    if not stripped:
        return 0.0
    letters = len(_LETTER_RE.findall(stripped))
    spaces = stripped.count(" ")
    alnum = sum(ch.isalnum() for ch in stripped)
    if alnum == 0:
        return 0.05
    if letters < MIN_NAME_LETTERS:
        return 0.30
    ratio = (letters + spaces) / len(stripped)
    score = 1.0
    if ratio < MIN_NAME_LETTER_RATIO:
        score *= max(0.1, ratio / MIN_NAME_LETTER_RATIO)
    # Mild length bonus: saturates at 20 letters (most card names are shorter).
    score *= 0.85 + 0.15 * min(letters, 20) / 20.0
    return score


def set_code_plausibility(text: str) -> float:
    """1.0 when the text has a set-code shape (or can be coerced to one), else 0.0."""
    if looks_like_set_code(text):
        return 1.0
    return 1.0 if set_code_variants(text) else 0.0


def _distinct(readings: Sequence[Reading]) -> list[Reading]:
    """Keep the highest-confidence reading per distinct (stripped) text, drop empties."""
    best: dict[str, Reading] = {}
    for variant, text, confidence, raw in readings:
        key = text.strip()
        if not key:
            continue
        current = best.get(key)
        if current is None or confidence > current[2]:
            best[key] = (variant, key, float(confidence), raw)
    return list(best.values())


def _rank_key(kind: ReadingKind, reading: Reading) -> tuple[float, float, int]:
    _, text, confidence, _ = reading
    if kind == "set_code":
        return (set_code_plausibility(text), confidence, len(text))
    return (confidence * name_plausibility(text), confidence, len(text))


def select_best_reading(readings: Sequence[Reading], kind: ReadingKind) -> OCRResult:
    """Choose the most plausible reading for ``kind`` and attach the rest as alternatives.

    * ``"set_code"``: readings that look like a set code (``looks_like_set_code``
      or a non-empty ``set_code_variants``) win; among them the highest
      confidence.  Without any plausible reading the highest confidence wins.
    * ``"name"``: rank by ``confidence * name_plausibility(text)``.

    Returns :meth:`OCRResult.empty` when there is no non-empty reading.
    """
    if kind not in ("name", "set_code"):
        raise ValueError(f"Unknown reading kind {kind!r}")
    distinct = _distinct(readings)
    if not distinct:
        return OCRResult.empty()
    ordered = sorted(distinct, key=lambda r: _rank_key(kind, r), reverse=True)
    variant, text, confidence, raw = ordered[0]
    alternatives = [
        OCRAlternative(text=alt_text, confidence=alt_conf, variant=alt_variant)
        for alt_variant, alt_text, alt_conf, _ in ordered[1:]
    ]
    return OCRResult(text=text, confidence=confidence, raw_result=raw, variant=variant, alternatives=alternatives)
