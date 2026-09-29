import CameraCapture from "./CameraCapture";
import ErrorAlert from "./ErrorAlert";
import Spinner from "./Spinner";
import UploadPicker from "./UploadPicker";

export type ScanMode = "camera" | "upload";

interface ScannerScreenProps {
  mode: ScanMode;
  onModeChange: (mode: ScanMode) => void;
  /** True while a recognition request is in flight. */
  busy: boolean;
  /** API error to show inline (null when there is none). */
  error: string | null;
  onSubmit: (blob: Blob) => void;
  /** Re-sends the last image; undefined when there is nothing to retry. */
  onRetry?: () => void;
  onDismissError: () => void;
  /** Ask the API for debug images (developer option). */
  debug: boolean;
  onDebugChange: (value: boolean) => void;
}

const MODES: { id: ScanMode; label: string }[] = [
  { id: "camera", label: "Use camera" },
  { id: "upload", label: "Upload image" },
];

/** Scanner screen: mode switch (camera / upload), the active capture panel and the busy overlay. */
export default function ScannerScreen({
  mode,
  onModeChange,
  busy,
  error,
  onSubmit,
  onRetry,
  onDismissError,
  debug,
  onDebugChange,
}: ScannerScreenProps) {
  return (
    <section className="stack" aria-busy={busy}>
      <div className="tabs" role="tablist" aria-label="Image source">
        {MODES.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            id={`tab-${item.id}`}
            aria-selected={mode === item.id}
            aria-controls="scanner-panel"
            className="btn tab"
            onClick={() => onModeChange(item.id)}
            disabled={busy}
          >
            {item.label}
          </button>
        ))}
      </div>

      {error && <ErrorAlert message={error} onRetry={onRetry} onDismiss={onDismissError} />}

      <div id="scanner-panel" role="tabpanel" aria-labelledby={`tab-${mode}`}>
        {mode === "camera" ? (
          <CameraCapture key="camera" disabled={busy} onSubmit={onSubmit} />
        ) : (
          <UploadPicker key="upload" disabled={busy} onSubmit={onSubmit} />
        )}
      </div>

      <label className="check">
        <input
          type="checkbox"
          checked={debug}
          onChange={(event) => onDebugChange(event.target.checked)}
          disabled={busy}
        />
        Request debug images from the API (developer option)
      </label>

      {busy && (
        <div className="busy" role="status" aria-live="polite">
          <div className="busy__card">
            <Spinner />
            <span>Recognizing...</span>
          </div>
        </div>
      )}
    </section>
  );
}
