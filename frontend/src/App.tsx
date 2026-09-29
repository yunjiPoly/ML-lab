import { useCallback, useRef, useState } from "react";
import { errorMessage, recognizeImage } from "./api/client";
import ErrorBoundary from "./components/ErrorBoundary";
import HealthIndicator from "./components/HealthIndicator";
import ResultCard from "./components/ResultCard";
import ScannerScreen, { type ScanMode } from "./components/ScannerScreen";
import type { RecognitionResult } from "./types/recognition";

const APP_NAME = "Yu-Gi-Oh! Card Scanner";

type Screen =
  | { kind: "scanner" }
  | { kind: "result"; result: RecognitionResult; previewUrl: string };

/**
 * Application shell: header (name + API health), the scanner or the result
 * screen, and the recognition request state shared between them.
 */
export default function App() {
  const [mode, setMode] = useState<ScanMode>("camera");
  const [debug, setDebug] = useState(false);
  const [screen, setScreen] = useState<Screen>({ kind: "scanner" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const lastBlobRef = useRef<Blob | null>(null);

  const submit = useCallback(
    async (blob: Blob) => {
      lastBlobRef.current = blob;
      setBusy(true);
      setError(null);
      try {
        const result = await recognizeImage(blob, { debug });
        setScreen({ kind: "result", result, previewUrl: URL.createObjectURL(blob) });
      } catch (err) {
        setError(errorMessage(err));
      } finally {
        setBusy(false);
      }
    },
    [debug],
  );

  const retry = useCallback(() => {
    const blob = lastBlobRef.current;
    if (blob) {
      void submit(blob);
    }
  }, [submit]);

  const reset = useCallback(() => {
    if (screen.kind === "result") {
      URL.revokeObjectURL(screen.previewUrl);
    }
    lastBlobRef.current = null;
    setError(null);
    setScreen({ kind: "scanner" });
  }, [screen]);

  return (
    <div className="app">
      <header className="app__header">
        <div className="app__brand">
          <span className="app__logo" aria-hidden="true" />
          <h1 className="app__title">{APP_NAME}</h1>
        </div>
        <HealthIndicator />
      </header>

      <main className="app__main">
        <ErrorBoundary>
          {screen.kind === "scanner" ? (
            <ScannerScreen
              mode={mode}
              onModeChange={setMode}
              busy={busy}
              error={error}
              onSubmit={(blob) => void submit(blob)}
              onRetry={lastBlobRef.current ? retry : undefined}
              onDismissError={() => setError(null)}
              debug={debug}
              onDebugChange={setDebug}
            />
          ) : (
            <ResultCard result={screen.result} previewUrl={screen.previewUrl} onReset={reset} />
          )}
        </ErrorBoundary>
      </main>

      <footer className="app__footer">
        Photos are sent only to the recognition API and are not stored in the browser.
      </footer>
    </div>
  );
}
