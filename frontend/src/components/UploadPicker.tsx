import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { formatBytes } from "../utils/format";

/** Mirrors the backend default `MAX_UPLOAD_BYTES` (12 MiB). */
const MAX_UPLOAD_BYTES = 12 * 1024 * 1024;

interface Selected {
  file: File;
  url: string;
}

interface UploadPickerProps {
  disabled: boolean;
  onSubmit: (blob: Blob) => void;
}

function validate(file: File): string | null {
  if (!file.type.startsWith("image/")) {
    return "Please choose an image file (JPEG, PNG or WebP).";
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return `The image is ${formatBytes(file.size)}; the limit is ${formatBytes(MAX_UPLOAD_BYTES)}.`;
  }
  return null;
}

/**
 * File picker (with `capture="environment"` so phones open the camera app),
 * a drag-and-drop zone for desktops, and a preview with a Recognize button.
 */
export default function UploadPicker({ disabled, onSubmit }: UploadPickerProps) {
  const captureInputRef = useRef<HTMLInputElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [selected, setSelected] = useState<Selected | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  useEffect(
    () => () => {
      if (selected) {
        URL.revokeObjectURL(selected.url);
      }
    },
    [selected],
  );

  const choose = useCallback((file: File | undefined) => {
    if (!file) {
      return;
    }
    const problem = validate(file);
    if (problem) {
      setError(problem);
      return;
    }
    setError(null);
    setSelected({ file, url: URL.createObjectURL(file) });
  }, []);

  const onInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    choose(event.target.files?.[0]);
    // Allow picking the same file again after "Choose a different image".
    event.target.value = "";
  };

  const onDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (!dragOver && !disabled) {
      setDragOver(true);
    }
  };

  const onDragLeave = () => setDragOver(false);

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOver(false);
    if (!disabled) {
      choose(event.dataTransfer.files?.[0]);
    }
  };

  return (
    <div className="stack">
      <input
        ref={captureInputRef}
        className="visually-hidden"
        type="file"
        accept="image/*"
        capture="environment"
        onChange={onInputChange}
        disabled={disabled}
        tabIndex={-1}
        aria-hidden="true"
      />
      <input
        ref={fileInputRef}
        className="visually-hidden"
        type="file"
        accept="image/*"
        onChange={onInputChange}
        disabled={disabled}
        tabIndex={-1}
        aria-hidden="true"
      />

      {selected ? (
        <div className="preview">
          <img className="preview__img" src={selected.url} alt="Preview of the selected image" />
          <p className="preview__meta">
            {selected.file.name} - {formatBytes(selected.file.size)}
          </p>
          <div className="actions">
            <button type="button" className="btn" onClick={() => setSelected(null)} disabled={disabled}>
              Choose a different image
            </button>
            <button
              type="button"
              className="btn btn--primary"
              onClick={() => onSubmit(selected.file)}
              disabled={disabled}
            >
              Recognize
            </button>
          </div>
        </div>
      ) : (
        <div
          className={`dropzone${dragOver ? " dropzone--active" : ""}`}
          onDragEnter={onDragOver}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
        >
          <p>Drop a photo of the card here</p>
          <p className="dropzone__hint">or</p>
          <div className="actions">
            <button
              type="button"
              className="btn btn--primary"
              onClick={() => captureInputRef.current?.click()}
              disabled={disabled}
            >
              Take a photo
            </button>
            <button type="button" className="btn" onClick={() => fileInputRef.current?.click()} disabled={disabled}>
              Choose an image
            </button>
          </div>
          <p className="dropzone__hint">JPEG, PNG or WebP, up to {formatBytes(MAX_UPLOAD_BYTES)}.</p>
        </div>
      )}

      {error && (
        <p className="note note--error" role="alert">
          {error}
        </p>
      )}
      <p className="note">The image is only sent to the recognition API; nothing is kept in the browser beyond this preview.</p>
    </div>
  );
}
