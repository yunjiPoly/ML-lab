/** Small formatting helpers shared by the UI components. */

/** Formats a 0..1 score as a whole percentage, e.g. `0.873 -> "87%"`. */
export function percent(value: number): string {
  if (!Number.isFinite(value)) {
    return "-";
  }
  const clamped = Math.min(Math.max(value, 0), 1);
  return `${Math.round(clamped * 100)}%`;
}

/** Formats a millisecond duration, e.g. `12.345 -> "12.3 ms"`. */
export function formatMs(value: number): string {
  return Number.isFinite(value) ? `${value.toFixed(1)} ms` : "-";
}

/** Formats a byte count for humans (B / KB / MB). */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "-";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${Math.round(bytes / 1024)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
