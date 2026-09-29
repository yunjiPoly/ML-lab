import type { RecognitionStatus } from "../types/recognition";

export type BadgeTone = "success" | "warning" | "danger" | "neutral";

export interface StatusMeta {
  label: string;
  tone: BadgeTone;
}

export const STATUS_META: Record<RecognitionStatus, StatusMeta> = {
  MATCHED: { label: "Matched", tone: "success" },
  LOW_CONFIDENCE: { label: "Low confidence - please verify", tone: "warning" },
  AMBIGUOUS: { label: "Ambiguous - pick the right one", tone: "warning" },
  NOT_FOUND: { label: "No matching card", tone: "neutral" },
  CARD_NOT_DETECTED: { label: "No card detected - try again with the card filling the guide", tone: "danger" },
  OCR_FAILED: { label: "Could not read text", tone: "danger" },
};

/** Label + tone for a status; unknown values (future statuses) fall back to a neutral badge. */
export function statusMeta(status: string): StatusMeta {
  const known = (STATUS_META as Record<string, StatusMeta | undefined>)[status];
  return known ?? { label: status, tone: "neutral" };
}

export default function StatusBadge({ status }: { status: RecognitionStatus }) {
  const { label, tone } = statusMeta(status);
  return (
    <span className={`badge badge--${tone}`} role="status">
      {label}
    </span>
  );
}
