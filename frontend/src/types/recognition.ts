/**
 * TypeScript mirror of the backend JSON contract for `POST /api/recognize`
 * (see `backend/app/schemas/recognition.py`). Keep these in sync with the
 * Pydantic models; field names are intentionally snake_case like the wire format.
 */

export type RecognitionStatus =
  | "MATCHED"
  | "LOW_CONFIDENCE"
  | "AMBIGUOUS"
  | "NOT_FOUND"
  | "CARD_NOT_DETECTED"
  | "OCR_FAILED";

export const RECOGNITION_STATUSES: readonly RecognitionStatus[] = [
  "MATCHED",
  "LOW_CONFIDENCE",
  "AMBIGUOUS",
  "NOT_FOUND",
  "CARD_NOT_DETECTED",
  "OCR_FAILED",
];

export interface OcrFieldResult {
  /** Text exactly as returned by the OCR engine. */
  raw: string;
  /** Canonical form used for the database lookup. */
  normalized: string;
  /** OCR engine score for the chosen reading, in [0, 1]. */
  confidence: number;
  /** Preprocessing variant that produced the reading. */
  variant: string | null;
  /** Other readings that were considered. */
  alternatives: string[];
}

export interface CardSummary {
  id: number;
  name: string;
  type: string | null;
  frame_type: string | null;
}

export interface PrintingSummary {
  id: number | null;
  set_code: string;
  set_name: string;
  rarity: string | null;
  rarity_code: string | null;
}

export interface Candidate {
  card: CardSummary;
  printing: PrintingSummary | null;
  /** Application-level match score in [0, 1] (not calibrated). */
  score: number;
  reasons: string[];
}

export interface DetectionInfo {
  detected: boolean;
  method: string | null;
  /** Ordered card corners in source-image pixels: TL, TR, BR, BL. */
  corners: number[][] | null;
  /** Degrees the normalized card was rotated to be upright. */
  rotation_applied: number;
  /** ROI template used. */
  layout: string;
}

export interface RecognitionOcr {
  name: OcrFieldResult | null;
  set_code: OcrFieldResult | null;
}

export interface RecognitionResult {
  status: RecognitionStatus;
  card: CardSummary | null;
  printing: PrintingSummary | null;
  ocr: RecognitionOcr;
  /** Application-level match score in [0, 1]. Not a calibrated probability. */
  confidence: number;
  candidates: Candidate[];
  /** Human-readable explanations / caveats. */
  notes: string[];
  detection: DetectionInfo | null;
  /** Paths of debug images when debug output is enabled. */
  debug: Record<string, string> | null;
  timing_ms: Record<string, number>;
}
