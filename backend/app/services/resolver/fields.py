"""Pure helpers of the resolver: OCR field parsing and ORM -> schema summaries.

Nothing here touches the database.  :func:`parse_name_field` and
:func:`parse_set_code_field` turn an :class:`OCRResult` into the
:class:`OcrFieldResult` reported to the client plus the internal data the
decision logic needs (normalized readings, ordered set-code variants, which
reading produced each variant).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.text import normalize_card_name, normalize_set_code, set_code_variants
from app.models import Card, Printing
from app.schemas.recognition import CardSummary, OcrFieldResult, PrintingSummary
from app.services.ocr.base import OCRResult


def clamp(value: float) -> float:
    """Clamp a score into ``[0, 1]``."""
    return max(0.0, min(1.0, float(value)))


def card_summary(card: Card) -> CardSummary:
    """ORM ``Card`` -> API summary."""
    return CardSummary(id=card.id, name=card.name, type=card.type, frame_type=card.frame_type)


def printing_summary(printing: Printing) -> PrintingSummary:
    """ORM ``Printing`` -> API summary."""
    return PrintingSummary(
        id=printing.id,
        set_code=printing.set_code,
        set_name=printing.set_name,
        rarity=printing.rarity,
        rarity_code=printing.rarity_code,
    )


@dataclass(frozen=True)
class NameField:
    """Parsed name OCR field."""

    result: OcrFieldResult | None
    readings: list[tuple[str, float]] = field(default_factory=list)  # (normalized text, confidence); selected first
    usable: bool = False

    @property
    def normalized(self) -> str:
        return self.readings[0][0] if self.readings else ""


@dataclass(frozen=True)
class SetCodeField:
    """Parsed set-code OCR field with the ordered variants derived from all readings."""

    result: OcrFieldResult | None
    variants: list[str] = field(default_factory=list)
    sources: dict[str, tuple[str, float]] = field(default_factory=dict)  # variant -> (reading, confidence)
    exact_codes: frozenset[str] = frozenset()  # plain normalized forms of the readings
    usable: bool = False

    @property
    def raw(self) -> str:
        return self.result.raw.strip() if self.result else ""

    def source_of(self, variant: str) -> tuple[str, float]:
        """``(reading text, reading confidence)`` that produced ``variant``."""
        return self.sources.get(variant, (self.raw, 0.0))

    def select(self, matched: str) -> None:
        """Record the code that matched the database: it becomes ``normalized`` and
        every other variant tried (including the as-is reading) is listed as alternative."""
        if self.result is None:
            return
        self.result.normalized = matched
        self.result.alternatives = [v for v in self.variants if v != matched]


def parse_name_field(ocr: OCRResult | None, *, min_confidence: float) -> NameField:
    """Normalize the name reading and its alternatives.

    The field is *usable* when the selected reading has text and its confidence
    is at least ``min_confidence``.
    """
    if ocr is None:
        return NameField(result=None)
    raw = ocr.text or ""
    confidence = clamp(ocr.confidence)
    selected = normalize_card_name(raw)
    readings: list[tuple[str, float]] = [(selected, confidence)] if selected else []
    for text, score in ocr.all_readings():
        normalized = normalize_card_name(text)
        if normalized and all(normalized != known for known, _ in readings):
            readings.append((normalized, clamp(score)))
    result = OcrFieldResult(
        raw=raw,
        normalized=selected,
        confidence=confidence,
        variant=ocr.variant,
        alternatives=[text for text, _ in readings if text != selected],
    )
    return NameField(result=result, readings=readings, usable=bool(selected) and confidence >= min_confidence)


def parse_set_code_field(ocr: OCRResult | None, *, min_confidence: float) -> SetCodeField:
    """Expand the set-code reading and its alternatives into ordered lookup variants.

    Variants of the selected reading come first (as-is first), then those of the
    alternative readings.  The plain normalized form of every reading is tried
    even when it does not have a canonical set-code shape.  The field is *usable*
    when it has text and either its confidence reaches ``min_confidence`` or at
    least one variant has a set-code shape (the database validates it).
    """
    if ocr is None:
        return SetCodeField(result=None)
    raw = ocr.text or ""
    confidence = clamp(ocr.confidence)
    readings: list[tuple[str, float]] = [(raw.strip(), confidence)] if raw.strip() else []
    readings += [(text, clamp(score)) for text, score in ocr.all_readings() if text != raw.strip()]

    variants: list[str] = []
    sources: dict[str, tuple[str, float]] = {}
    exact_codes: set[str] = set()
    shaped = False
    for text, score in readings:
        plain = normalize_set_code(text)
        derived = set_code_variants(text)
        shaped = shaped or bool(derived)
        if plain:
            exact_codes.add(plain)
        for variant in [*derived, plain]:
            if not variant:
                continue
            if variant not in sources or variant == plain:  # an as-is reading owns its own variant
                sources[variant] = (text, score)
            if variant not in variants:
                variants.append(variant)

    normalized = variants[0] if variants else normalize_set_code(raw)
    result = OcrFieldResult(
        raw=raw,
        normalized=normalized,
        confidence=confidence,
        variant=ocr.variant,
        alternatives=[v for v in variants if v != normalized],
    )
    usable = bool(readings) and (confidence >= min_confidence or shaped)
    return SetCodeField(result=result, variants=variants, sources=sources, exact_codes=frozenset(exact_codes), usable=usable)


def usability_notes(name: NameField, code: SetCodeField, *, min_confidence: float) -> list[str]:
    """Human-readable notes explaining why a provided field was ignored."""
    notes: list[str] = []
    if name.result is not None and not name.usable:
        if not name.normalized:
            notes.append("name OCR returned no text")
        else:
            notes.append(f"name OCR confidence {name.result.confidence:.2f} below minimum {min_confidence:.2f}; ignored")
    if code.result is not None and not code.usable:
        if not code.raw:
            notes.append("set code OCR returned no text")
        else:
            notes.append(
                f"set code OCR confidence {code.result.confidence:.2f} below minimum {min_confidence:.2f} "
                "and no reading looks like a set code; ignored"
            )
    return notes
