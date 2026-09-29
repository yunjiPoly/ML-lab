import type { RecognitionResult } from "../types/recognition";
import { percent } from "../utils/format";
import CandidateList from "./CandidateList";
import DeveloperDetails from "./DeveloperDetails";
import StatusBadge from "./StatusBadge";

interface ResultCardProps {
  result: RecognitionResult;
  /** Object URL of the scanned photo (thumbnail); null when unavailable. */
  previewUrl: string | null;
  /** Back to the scanner, keeping the chosen mode. */
  onReset: () => void;
}

const SCORE_HINT =
  "Application-level score combining OCR scores and database agreement. It is a ranking signal, not a calibrated probability.";

/** Recognition result: status badge, card / printing details, notes, candidates and developer details. */
export default function ResultCard({ result, previewUrl, onReset }: ResultCardProps) {
  const { card, printing, ocr } = result;
  const ambiguous = result.status === "AMBIGUOUS";
  const ocrName = ocr.name?.normalized || ocr.name?.raw || "";
  const ocrSetCode = ocr.set_code?.normalized || ocr.set_code?.raw || "";

  const heading = card?.name ?? (ocrName ? `Read as "${ocrName}"` : "Unknown card");

  return (
    <section className="stack" aria-labelledby="result-title">
      <div className="card stack">
        <div className="result__head">
          <StatusBadge status={result.status} />
          {previewUrl && <img className="result__thumb" src={previewUrl} alt="The photo that was scanned" />}
        </div>

        {ambiguous && <CandidateList candidates={result.candidates} primary title="Pick the right card" />}

        <div className="stack">
          <h2 id="result-title" className="result__name">
            {heading}
          </h2>

          {printing ? (
            <div className="setcode" aria-label={`Set code ${printing.set_code}`}>
              {printing.set_code}
            </div>
          ) : ocrSetCode ? (
            <div className="setcode setcode--muted" title="Set code as read by OCR (no printing matched)">
              {ocrSetCode}
            </div>
          ) : null}

          <dl className="kv-inline">
            {printing && (
              <>
                <dt>Set</dt>
                <dd>{printing.set_name || "-"}</dd>
                <dt>Rarity</dt>
                <dd>
                  {printing.rarity ?? "Unknown"}
                  {printing.rarity_code ? ` (${printing.rarity_code})` : ""}
                </dd>
              </>
            )}
            {card?.type && (
              <>
                <dt>Type</dt>
                <dd>
                  {card.type}
                  {card.frame_type ? ` / ${card.frame_type}` : ""}
                </dd>
              </>
            )}
          </dl>

          <p className="score">
            <span>Match score</span>
            <span className="score__value">{percent(result.confidence)}</span>
            <span className="score__hint" title={SCORE_HINT}>
              application score, not a probability
            </span>
          </p>
        </div>

        {result.notes.length > 0 && (
          <ul className="notes" aria-label="Notes">
            {result.notes.map((note, index) => (
              <li key={`${index}-${note}`}>{note}</li>
            ))}
          </ul>
        )}
      </div>

      {!ambiguous && <CandidateList candidates={result.candidates} />}

      <div className="actions">
        <button type="button" className="btn btn--primary btn--block" onClick={onReset}>
          Scan another card
        </button>
      </div>

      <DeveloperDetails result={result} />
    </section>
  );
}
