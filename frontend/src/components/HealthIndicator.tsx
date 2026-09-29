import { useCallback, useEffect, useState } from "react";
import { errorMessage, getHealth } from "../api/client";
import type { HealthResponse } from "../types/cards";

type HealthState =
  | { kind: "checking" }
  | { kind: "ok"; info: HealthResponse }
  | { kind: "error"; message: string };

/**
 * Tiny API health indicator for the header: calls `GET /api/health` on mount,
 * shows a green dot when the API answers and a red dot with the error otherwise.
 * Tapping it re-checks.
 */
export default function HealthIndicator() {
  const [state, setState] = useState<HealthState>({ kind: "checking" });

  const check = useCallback(async (signal?: AbortSignal) => {
    setState({ kind: "checking" });
    try {
      const info = await getHealth(signal);
      setState({ kind: "ok", info });
    } catch (err) {
      if (signal?.aborted) {
        return;
      }
      setState({ kind: "error", message: errorMessage(err) });
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void check(controller.signal);
    return () => controller.abort();
  }, [check]);

  const label =
    state.kind === "ok" ? "API online" : state.kind === "error" ? "API unreachable" : "Checking API...";
  const title =
    state.kind === "ok"
      ? `Backend healthy${state.info.status ? ` (status: ${String(state.info.status)})` : ""}. Tap to re-check.`
      : state.kind === "error"
        ? `${state.message}. Tap to retry.`
        : "Checking the API...";

  return (
    <>
      <button
        type="button"
        className={`health health--${state.kind}`}
        onClick={() => void check()}
        title={title}
        aria-live="polite"
        aria-label={`${label}. Tap to re-check.`}
      >
        <span className="health__dot" aria-hidden="true" />
        <span>{label}</span>
      </button>
      {state.kind === "error" && <span className="health__error">{state.message}</span>}
    </>
  );
}
