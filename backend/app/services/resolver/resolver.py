"""Resolver: OCR name + OCR set code + database -> :class:`ResolutionResult`.

Inputs are the two :class:`OCRResult` fields (either may be ``None``), optional
visual candidates, and the thresholds from :class:`Settings`
(``resolver_name_candidate_threshold`` 0.70, ``resolver_name_strong_threshold``
0.90, ``resolver_ambiguity_margin`` 0.05, ``resolver_max_candidates`` 5,
``ocr_min_confidence`` 0.30).

Decision logic
--------------
1. **Normalize.** The name goes through :func:`normalize_card_name`.  For the set
   code, the selected reading and every alternative reading are expanded with
   :func:`set_code_variants` (ordered, as-is first).  Both fields are reported
   back as :class:`OcrFieldResult` (raw, normalized, confidence, variant,
   alternatives tried).  See :mod:`app.services.resolver.fields`.
2. **OCR_FAILED** when both fields are unusable: empty text, or below
   ``ocr_min_confidence`` (a low-confidence set code is still usable when one of
   its variants has a set-code shape, because the database validates it).
3. **Name candidates** come from :class:`NameIndex` (rapidfuzz ratio /
   token_sort_ratio, exact normalized match = 1.0) with
   ``score_cutoff = candidate_threshold``.  Alternative readings are tried when
   the selected reading yields no strong candidate.
4. **Set-code candidates** are the printings whose normalized code equals any
   variant; ranked exact-as-is first, then by variant order (fewer corrections
   first).  The variant that matched is written to ``set_code.normalized``.
5. **Decide.**
   a) *Set-code hit(s).*  Grouped by card.  Name agrees (set-code card is the top
      name candidate, or ``name_similarity >= candidate_threshold``) ->
      ``MATCHED``.  Name disagrees but the code was an exact as-is hit read with
      confidence >= 0.85 -> ``MATCHED`` with reduced confidence and a note.
      Name disagrees (or is unusable) and the code only matched via a corrected
      variant (or was read with low confidence) -> ``LOW_CONFIDENCE`` with the
      set-code card as primary suggestion.  Hits on several different cards ->
      the one the name agrees with, else ``AMBIGUOUS``.  Several rarities of one
      card share the code -> ``MATCHED`` with the first printing, all rarities
      listed as candidates, and a note.
   b) *No set-code hit, name candidates.*  Top >= strong_threshold and the
      runner-up below ``top - ambiguity_margin`` (or top is an exact match) ->
      the card is identified.  Its printing codes are compared with the OCR
      variants using normalized Levenshtein similarity; a unique best >= 0.75 ->
      ``MATCHED`` (reduced confidence, note "set code matched fuzzily").  Else
      ``LOW_CONFIDENCE`` with the card, printing ``None`` (or the single known
      printing, flagged as assumed) and the printings as candidates.
      Strong but a runner-up within the margin -> ``AMBIGUOUS``.  Top between the
      thresholds -> ``LOW_CONFIDENCE`` with candidates.
   c) *Nothing.*  Visual candidates (if any) become ``LOW_CONFIDENCE``
      candidates, otherwise ``NOT_FOUND``.
6. **Confidence** is an application-level score, NOT a calibrated probability:
   exact set code + name agreement: ``0.6*code_conf + 0.3*name_similarity + 0.1``
   (capped at 1.0); corrected-variant matches x0.9; name disagreement x0.8; fuzzy
   set code: ``0.6*code_conf*code_similarity + 0.3*name_similarity + 0.1`` x0.9;
   name-only: ``name_similarity * name_ocr_confidence`` x0.8 when the card is
   identified but its printing is not, x0.5 when the card itself is uncertain.
   Every candidate carries its own score and reasons.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import Settings
from app.core.text import normalize_set_code
from app.db.protocols import CardRepository
from app.models import Card, Printing
from app.schemas.recognition import Candidate, PrintingSummary, RecognitionStatus, ResolutionResult
from app.services.ocr.base import OCRResult
from app.services.resolver.fields import (
    NameField,
    SetCodeField,
    card_summary,
    clamp,
    parse_name_field,
    parse_set_code_field,
    printing_summary,
    usability_notes,
)
from app.services.resolver.matching import closest_set_code, name_similarity
from app.services.resolver.name_index import NameIndex, NameMatch
from app.services.visual.base import VisualCandidate

logger = logging.getLogger(__name__)

TRUSTED_SET_CODE_CONFIDENCE = 0.85
"""An exact as-is set-code hit read at least this confidently overrides a disagreeing name."""
FUZZY_SET_CODE_MIN_SIMILARITY = 0.75
"""Minimum Levenshtein similarity for a printing code to be accepted for an identified card."""
CORRECTED_VARIANT_FACTOR = 0.9
NAME_DISAGREEMENT_FACTOR = 0.8
PRINTING_UNKNOWN_FACTOR = 0.8
CARD_UNCERTAIN_FACTOR = 0.5


@dataclass(frozen=True)
class SetCodeHit:
    """A printing whose code equals one of the OCR variants."""

    printing: Printing
    variant: str
    exact: bool
    rank: int


@dataclass(frozen=True)
class NameSearch:
    """Name-index result together with the reading that produced it."""

    matches: list[NameMatch]
    query: str
    confidence: float


def _code_score(code_conf: float, similarity: float, *, exact: bool) -> float:
    """Application-level score for a set-code hit (see module docstring, item 6)."""
    score = min(1.0, 0.6 * code_conf + 0.3 * similarity + 0.1)
    return score if exact else score * CORRECTED_VARIANT_FACTOR


class Resolver:
    """Combines OCR readings with the card database (see module docstring)."""

    def __init__(self, repository: CardRepository, settings: Settings, *, name_index: NameIndex | None = None) -> None:
        self._repo = repository
        self._candidate_threshold = settings.resolver_name_candidate_threshold
        self._strong_threshold = settings.resolver_name_strong_threshold
        self._margin = settings.resolver_ambiguity_margin
        self._max_candidates = max(1, settings.resolver_max_candidates)
        self._min_conf = settings.ocr_min_confidence
        self._index = name_index if name_index is not None else NameIndex.from_repository(repository)

    @property
    def name_index(self) -> NameIndex:
        return self._index

    # ------------------------------------------------------------------ api
    def resolve(
        self,
        name_ocr: OCRResult | None,
        set_code_ocr: OCRResult | None,
        *,
        visual_candidates: Sequence[VisualCandidate] = (),
    ) -> ResolutionResult:
        """Resolve one card from its OCR readings (never raises for bad input)."""
        name = parse_name_field(name_ocr, min_confidence=self._min_conf)
        code = parse_set_code_field(set_code_ocr, min_confidence=self._min_conf)
        notes = usability_notes(name, code, min_confidence=self._min_conf)

        if not name.usable and not code.usable:
            notes.append("OCR produced no usable name or set code")
            return ResolutionResult(
                status=RecognitionStatus.OCR_FAILED, name=name.result, set_code=code.result, confidence=0.0, notes=notes
            )

        search = self._search_names(name) if name.usable else NameSearch([], "", 0.0)
        hits = self._find_set_code_hits(code) if code.usable else []
        logger.debug("resolve: name=%r name_matches=%d set_code_hits=%d", name.normalized, len(search.matches), len(hits))

        if hits:
            result = self._resolve_by_set_code(hits, name, code, search)
        elif search.matches:
            result = self._resolve_by_name(search, name, code)
        else:
            result = self._resolve_fallback(name, code, visual_candidates)
        result.notes = [*notes, *result.notes]
        result.candidates = result.candidates[: self._max_candidates]
        return result

    # ------------------------------------------------------------- lookups
    def _search_names(self, name: NameField) -> NameSearch:
        best = NameSearch([], name.normalized, name.readings[0][1] if name.readings else 0.0)
        for reading, confidence in name.readings:
            if confidence < self._min_conf:
                continue
            matches = self._index.search(reading, limit=10, score_cutoff=self._candidate_threshold)
            if matches and (not best.matches or matches[0].score > best.matches[0].score):
                best = NameSearch(matches, reading, confidence)
            if best.matches and best.matches[0].score >= self._strong_threshold:
                break
        return best

    def _find_set_code_hits(self, code: SetCodeField) -> list[SetCodeHit]:
        if not code.variants:
            return []
        order = {variant: rank for rank, variant in enumerate(code.variants)}
        hits: list[SetCodeHit] = []
        for printing in self._repo.find_printings_by_set_codes(code.variants):
            stored = normalize_set_code(printing.set_code)
            if stored in order:
                hits.append(SetCodeHit(printing, stored, exact=stored in code.exact_codes, rank=order[stored]))
        hits.sort(key=lambda h: (not h.exact, h.rank, h.printing.card_id, h.printing.id or 0))
        return hits

    def _best_name_similarity(self, name: NameField, card: Card) -> float:
        if not name.usable:
            return 0.0
        scores = (name_similarity(reading, card.normalized_name) for reading, conf in name.readings if conf >= self._min_conf)
        return max(scores, default=0.0)

    def _name_candidates(self, matches: Sequence[NameMatch], confidence: float, *, exclude: int | None = None) -> list[Candidate]:
        candidates: list[Candidate] = []
        for match in matches:
            if match.card_id == exclude:
                continue
            card = self._repo.get_card(match.card_id)
            if card is None:
                logger.warning("Name index refers to unknown card id %s; refresh the index.", match.card_id)
                continue
            reasons = [f"name_similarity={match.score:.2f}", f"name_ocr_confidence={confidence:.2f}"]
            candidates.append(Candidate(card=card_summary(card), score=clamp(match.score * confidence), reasons=reasons))
            if len(candidates) >= self._max_candidates:
                break
        return candidates

    @staticmethod
    def _printing_candidates(card: Card, printings: Sequence[Printing], score: float, reasons: list[str]) -> list[Candidate]:
        """One candidate per printing; every entry after the first is flagged as a rarity alternative."""
        return [
            Candidate(
                card=card_summary(card),
                printing=printing_summary(p),
                score=clamp(score),
                reasons=reasons + (["rarity_alternative"] if i else []),
            )
            for i, p in enumerate(printings)
        ]

    # ------------------------------------------------------ branch a: code
    def _resolve_by_set_code(self, hits: list[SetCodeHit], name: NameField, code: SetCodeField, search: NameSearch) -> ResolutionResult:
        by_card: dict[int, list[SetCodeHit]] = {}
        for hit in hits:
            by_card.setdefault(hit.printing.card_id, []).append(hit)
        notes: list[str] = []
        if len(by_card) > 1:
            agreeing = [
                cid
                for cid, card_hits in by_card.items()
                if self._best_name_similarity(name, card_hits[0].printing.card) >= self._candidate_threshold
            ]
            if len(agreeing) != 1:
                return self._ambiguous_set_code(by_card, name, code)
            notes.append(f"set code variants matched {len(by_card)} cards; the name OCR selected one of them")
            card_hits = by_card[agreeing[0]]
        else:
            card_hits = next(iter(by_card.values()))

        best = card_hits[0]
        card = best.printing.card
        rarities = [h.printing for h in card_hits if h.variant == best.variant]
        reading, reading_conf = code.source_of(best.variant)
        similarity = self._best_name_similarity(name, card)
        agrees = (bool(search.matches) and search.matches[0].card_id == card.id) or similarity >= self._candidate_threshold
        assert code.result is not None  # a usable field always has a result
        code.select(best.variant)

        confidence = _code_score(reading_conf, similarity, exact=best.exact)
        reasons = ["set_code_exact" if best.exact else f"set_code_variant={best.variant}", f"name_similarity={similarity:.2f}"]
        if not best.exact:
            notes.append(f"set code corrected from OCR reading '{reading}' to '{best.variant}'")
        if code.raw and reading != code.raw:
            notes.append(f"set code taken from alternative OCR reading '{reading}'")

        if agrees:
            status = RecognitionStatus.MATCHED
        elif best.exact and reading_conf >= TRUSTED_SET_CODE_CONFIDENCE:
            status = RecognitionStatus.MATCHED
            confidence *= NAME_DISAGREEMENT_FACTOR
            notes.append(("name OCR disagreed" if name.usable else "name OCR unusable") + "; trusting printed set code")
        else:
            status = RecognitionStatus.LOW_CONFIDENCE
            confidence *= NAME_DISAGREEMENT_FACTOR
            why = "name OCR disagreed" if name.usable else "name OCR unusable"
            how = (
                "set code only matched via a corrected variant"
                if not best.exact
                else f"set code OCR confidence {reading_conf:.2f} is low"
            )
            notes.append(f"{why} and {how}; verify manually")

        if len(rarities) > 1:
            notes.append(f"{len(rarities)} rarities share this set code; rarity could not be determined from text")
        candidates = self._printing_candidates(card, rarities, confidence, reasons)
        if not agrees:
            candidates += self._name_candidates(search.matches, search.confidence, exclude=card.id)
        return ResolutionResult(
            status=status, card=card_summary(card), printing=printing_summary(rarities[0]), name=name.result,
            set_code=code.result, confidence=clamp(confidence), candidates=candidates, notes=notes,
        )

    def _ambiguous_set_code(self, by_card: dict[int, list[SetCodeHit]], name: NameField, code: SetCodeField) -> ResolutionResult:
        candidates: list[Candidate] = []
        for card_hits in by_card.values():
            hit = card_hits[0]
            card = hit.printing.card
            similarity = self._best_name_similarity(name, card)
            score = _code_score(code.source_of(hit.variant)[1], similarity, exact=hit.exact)
            reasons = ["set_code_exact" if hit.exact else f"set_code_variant={hit.variant}", f"name_similarity={similarity:.2f}"]
            candidates.append(
                Candidate(card=card_summary(card), printing=printing_summary(hit.printing), score=clamp(score), reasons=reasons)
            )
        candidates.sort(key=lambda c: -c.score)
        return ResolutionResult(
            status=RecognitionStatus.AMBIGUOUS, name=name.result, set_code=code.result,
            confidence=clamp(candidates[0].score * CARD_UNCERTAIN_FACTOR), candidates=candidates,
            notes=[f"set code variants matched {len(by_card)} different cards and the name OCR could not decide"],
        )

    # ------------------------------------------------------ branch b: name
    def _resolve_by_name(self, search: NameSearch, name: NameField, code: SetCodeField) -> ResolutionResult:
        top = search.matches[0]
        runner = search.matches[1] if len(search.matches) > 1 else None
        notes: list[str] = []
        if search.query != name.normalized:
            notes.append(f"name taken from alternative OCR reading '{search.query}'")
        strong = top.score >= self._strong_threshold
        ambiguous = runner is not None and runner.score >= top.score - self._margin
        base = top.score * search.confidence

        if strong and (top.is_exact or not ambiguous):
            card = self._repo.get_card(top.card_id)
            if card is not None:
                result = self._identified_by_name(card, top, search, name, code)
                result.notes = [*notes, *result.notes]
                return result
            logger.warning("Name index refers to unknown card id %s; refresh the index.", top.card_id)

        if strong and ambiguous:
            close = [m for m in search.matches if m.score >= top.score - self._margin]
            notes.append(f"{len(close)} card names are within {self._margin:.2f} of the best match and no set code was matched")
            return ResolutionResult(
                status=RecognitionStatus.AMBIGUOUS,
                name=name.result,
                set_code=code.result,
                confidence=clamp(base * CARD_UNCERTAIN_FACTOR),
                candidates=self._name_candidates(close, search.confidence),
                notes=notes,
            )
        notes.append(
            f"best name match {top.score:.2f} is below the strong threshold {self._strong_threshold:.2f} "
            "and no set code was matched"
        )
        return ResolutionResult(
            status=RecognitionStatus.LOW_CONFIDENCE,
            name=name.result,
            set_code=code.result,
            confidence=clamp(base * CARD_UNCERTAIN_FACTOR),
            candidates=self._name_candidates(search.matches, search.confidence),
            notes=notes,
        )

    def _identified_by_name(self, card: Card, top: NameMatch, search: NameSearch, name: NameField, code: SetCodeField) -> ResolutionResult:
        printings = list(card.printings) or list(self._repo.printings_for_card(card.id))
        codes = [p.set_code for p in printings]
        name_reason = f"name_similarity={top.score:.2f}"
        notes: list[str] = []

        # Fuzzy set-code match against this card's own printings.
        best_variant, best_code, best_sim = "", None, 0.0
        for variant in code.variants if code.usable else []:
            matched, sim = closest_set_code(variant, codes)
            if matched is not None and sim > best_sim:
                best_variant, best_code, best_sim = variant, matched, sim
        if best_code is not None and best_sim >= FUZZY_SET_CODE_MIN_SIMILARITY:
            tied = {c for c in codes if closest_set_code(best_variant, [c])[1] >= best_sim - 1e-9}
            if tied == {best_code}:
                assert code.result is not None
                code.select(best_code)
                code_conf = code.source_of(best_variant)[1]
                rarities = [p for p in printings if p.set_code == best_code]
                confidence = min(1.0, 0.6 * code_conf * best_sim + 0.3 * top.score + 0.1) * CORRECTED_VARIANT_FACTOR
                notes.append(f"set code matched fuzzily ('{best_variant}' ~ '{best_code}', similarity {best_sim:.2f})")
                if len(rarities) > 1:
                    notes.append(f"{len(rarities)} rarities share this set code; rarity could not be determined from text")
                candidates = self._printing_candidates(card, rarities, confidence, [name_reason, f"set_code_similarity={best_sim:.2f}"])
                return ResolutionResult(
                    status=RecognitionStatus.MATCHED, card=card_summary(card), printing=printing_summary(rarities[0]), name=name.result,
                    set_code=code.result, confidence=clamp(confidence), candidates=candidates, notes=notes,
                )
            notes.append(f"several printings are equally close to the OCR'd set code '{best_variant}'")

        # Card identified, printing not: list the printings, closest to the OCR'd code first.
        def closeness(printing: Printing) -> float:
            return max((closest_set_code(v, [printing.set_code])[1] for v in code.variants), default=0.0)

        confidence = clamp(top.score * search.confidence * PRINTING_UNKNOWN_FACTOR)
        ranked = sorted(printings, key=lambda p: -closeness(p)) if code.variants else printings
        candidates = [
            Candidate(
                card=card_summary(card),
                printing=printing_summary(p),
                score=confidence,
                reasons=[name_reason, "printing_unverified"]
                + ([f"set_code_similarity={closeness(p):.2f}"] if code.variants else []),
            )
            for p in ranked[: self._max_candidates]
        ]
        printing: PrintingSummary | None = None
        if len(printings) == 1:
            printing = printing_summary(printings[0])
            notes.append("card has a single known printing; it is assumed, not read from the set code")
        else:
            notes.append(
                "set code could not be matched; printing undetermined"
                if code.usable
                else "set code unreadable; printing undetermined"
            )
        return ResolutionResult(
            status=RecognitionStatus.LOW_CONFIDENCE, card=card_summary(card), printing=printing, name=name.result,
            set_code=code.result, confidence=confidence, candidates=candidates, notes=notes,
        )

    # --------------------------------------------------- branch c: nothing
    def _resolve_fallback(self, name: NameField, code: SetCodeField, visual_candidates: Sequence[VisualCandidate]) -> ResolutionResult:
        candidates: list[Candidate] = []
        for visual in sorted(visual_candidates, key=lambda v: -v.score):
            card = self._repo.get_card(visual.card_id)
            if card is None:
                logger.debug("Visual candidate refers to unknown card id %s; skipped.", visual.card_id)
                continue
            candidates.append(
                Candidate(card=card_summary(card), score=clamp(visual.score), reasons=[visual.reason or "visual_similarity"])
            )
            if len(candidates) >= self._max_candidates:
                break
        if candidates:
            return ResolutionResult(
                status=RecognitionStatus.LOW_CONFIDENCE, name=name.result, set_code=code.result,
                confidence=clamp(candidates[0].score * CARD_UNCERTAIN_FACTOR), candidates=candidates,
                notes=["no text match; candidates come from visual similarity only"],
            )
        searched: list[str] = []
        if name.usable:
            searched.append(f"name '{name.normalized}'")
        if code.usable and code.result is not None:
            searched.append(f"set code '{code.result.normalized}'")
        return ResolutionResult(
            status=RecognitionStatus.NOT_FOUND, name=name.result, set_code=code.result, confidence=0.0,
            notes=[f"no card matched {' or '.join(searched)}"],
        )
