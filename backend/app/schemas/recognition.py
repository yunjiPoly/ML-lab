"""Recognition result schema (the JSON contract of ``POST /api/recognize``).

``confidence`` is an *application-level match score* in ``[0, 1]`` combining OCR
scores and database agreement.  It is NOT a calibrated probability.  Treat it
as a ranking / gating signal only.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RecognitionStatus(str, Enum):
    MATCHED = "MATCHED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_FOUND = "NOT_FOUND"
    CARD_NOT_DETECTED = "CARD_NOT_DETECTED"
    OCR_FAILED = "OCR_FAILED"


class OcrFieldResult(BaseModel):
    raw: str = Field(description="Text exactly as returned by the OCR engine.")
    normalized: str = Field(description="Canonical form used for database lookup.")
    confidence: float = Field(ge=0.0, le=1.0, description="OCR engine score for the chosen reading.")
    variant: str | None = Field(default=None, description="Preprocessing variant that produced the reading.")
    alternatives: list[str] = Field(
        default_factory=list,
        description="Other readings considered (other preprocessing variants / confusion corrections).",
    )


class CardSummary(BaseModel):
    id: int
    name: str
    type: str | None = None
    frame_type: str | None = None


class PrintingSummary(BaseModel):
    id: int | None = None
    set_code: str
    set_name: str
    rarity: str | None = None
    rarity_code: str | None = None


class Candidate(BaseModel):
    card: CardSummary
    printing: PrintingSummary | None = None
    score: float = Field(ge=0.0, le=1.0, description="Application-level match score (not calibrated).")
    reasons: list[str] = Field(default_factory=list)


class RecognitionOcr(BaseModel):
    name: OcrFieldResult | None = None
    set_code: OcrFieldResult | None = None


class DetectionInfo(BaseModel):
    detected: bool
    method: str | None = Field(default=None, description="Detection strategy that found the card.")
    corners: list[list[float]] | None = Field(
        default=None, description="Ordered card corners in source-image pixels: TL, TR, BR, BL."
    )
    rotation_applied: int = Field(default=0, description="Degrees the normalized card was rotated to be upright.")
    layout: str = Field(default="standard", description="ROI template used.")


class ResolutionResult(BaseModel):
    """Output of the resolver.  The pipeline copies these fields into the final result."""

    status: RecognitionStatus
    card: CardSummary | None = None
    printing: PrintingSummary | None = None
    name: OcrFieldResult | None = None
    set_code: OcrFieldResult | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    candidates: list[Candidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RecognitionResult(BaseModel):
    status: RecognitionStatus
    card: CardSummary | None = None
    printing: PrintingSummary | None = None
    ocr: RecognitionOcr = Field(default_factory=RecognitionOcr)
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Application-level match score in [0,1]. Not a calibrated probability.",
    )
    candidates: list[Candidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list, description="Human-readable explanations / caveats.")
    detection: DetectionInfo | None = None
    debug: dict[str, str] | None = Field(
        default=None, description="Paths of debug images when debug output is enabled."
    )
    timing_ms: dict[str, float] = Field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        return self.status == RecognitionStatus.MATCHED
