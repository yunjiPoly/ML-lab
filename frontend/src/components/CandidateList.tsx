import type { Candidate } from "../types/recognition";
import { percent } from "../utils/format";

interface CandidateListProps {
  candidates: Candidate[];
  /** Visually emphasise the list (used for AMBIGUOUS results). */
  primary?: boolean;
  title?: string;
}

function candidateKey(candidate: Candidate, index: number): string {
  const printing = candidate.printing?.id ?? candidate.printing?.set_code ?? "none";
  return `${candidate.card.id}-${printing}-${index}`;
}

/** Ordered list of candidate cards with set code, rarity and match score. */
export default function CandidateList({ candidates, primary = false, title = "Candidates" }: CandidateListProps) {
  if (candidates.length === 0) {
    return null;
  }
  return (
    <section className="stack" aria-label={title}>
      <h3 className="section-title">{title}</h3>
      <ol className={`candidates${primary ? " candidates--primary" : ""}`}>
        {candidates.map((candidate, index) => {
          const meta = [candidate.printing?.set_name, candidate.printing?.rarity].filter(Boolean).join(" - ");
          return (
            <li key={candidateKey(candidate, index)} className="candidate">
              <span className="candidate__name">{candidate.card.name}</span>
              <span className="candidate__score" aria-label={`Match score ${percent(candidate.score)}`}>
                {percent(candidate.score)}
              </span>
              <span className="candidate__code">{candidate.printing?.set_code ?? "No set code"}</span>
              <span className="candidate__meta">
                {meta || candidate.card.type || "Unknown printing"}
                {candidate.reasons.length > 0 ? ` | ${candidate.reasons.join("; ")}` : ""}
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
