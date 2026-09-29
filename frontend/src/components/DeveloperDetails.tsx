import { useMemo } from "react";
import type { DetectionInfo, OcrFieldResult, RecognitionResult } from "../types/recognition";
import { formatMs, percent } from "../utils/format";

function OcrTable({ label, field }: { label: string; field: OcrFieldResult | null }) {
  return (
    <table className="kv">
      <caption>{label}</caption>
      <tbody>
        {field ? (
          <>
            <tr>
              <th scope="row">Raw</th>
              <td>{field.raw || "(empty)"}</td>
            </tr>
            <tr>
              <th scope="row">Normalized</th>
              <td>{field.normalized || "(empty)"}</td>
            </tr>
            <tr>
              <th scope="row">Confidence</th>
              <td>{percent(field.confidence)}</td>
            </tr>
            <tr>
              <th scope="row">Variant</th>
              <td>{field.variant ?? "-"}</td>
            </tr>
            <tr>
              <th scope="row">Alternatives</th>
              <td>{field.alternatives.length > 0 ? field.alternatives.join(", ") : "-"}</td>
            </tr>
          </>
        ) : (
          <tr>
            <td colSpan={2}>Not available</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

function formatCorners(corners: number[][] | null): string {
  if (!corners || corners.length === 0) {
    return "-";
  }
  return corners.map((point) => `(${point.map((v) => Math.round(v)).join(", ")})`).join(" ");
}

function DetectionTable({ detection }: { detection: DetectionInfo | null }) {
  return (
    <table className="kv">
      <caption>Detection</caption>
      <tbody>
        {detection ? (
          <>
            <tr>
              <th scope="row">Detected</th>
              <td>{detection.detected ? "yes" : "no"}</td>
            </tr>
            <tr>
              <th scope="row">Method</th>
              <td>{detection.method ?? "-"}</td>
            </tr>
            <tr>
              <th scope="row">Rotation applied</th>
              <td>{detection.rotation_applied} deg</td>
            </tr>
            <tr>
              <th scope="row">Layout</th>
              <td>{detection.layout}</td>
            </tr>
            <tr>
              <th scope="row">Corners (TL, TR, BR, BL)</th>
              <td>{formatCorners(detection.corners)}</td>
            </tr>
          </>
        ) : (
          <tr>
            <td colSpan={2}>Not available</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

function TimingTable({ timing }: { timing: Record<string, number> }) {
  const entries = Object.entries(timing);
  return (
    <table className="kv">
      <caption>Timing</caption>
      <tbody>
        {entries.length > 0 ? (
          entries.map(([key, value]) => (
            <tr key={key}>
              <th scope="row">{key}</th>
              <td>{formatMs(value)}</td>
            </tr>
          ))
        ) : (
          <tr>
            <td colSpan={2}>No timing reported</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

/** Collapsed-by-default panel with OCR readings, detection info, timing and the raw JSON. */
export default function DeveloperDetails({ result }: { result: RecognitionResult }) {
  const json = useMemo(() => JSON.stringify(result, null, 2), [result]);
  const debugEntries = result.debug ? Object.entries(result.debug) : [];

  return (
    <details className="dev">
      <summary>Developer details</summary>
      <div className="dev__body">
        <h3>OCR</h3>
        <OcrTable label="Name" field={result.ocr.name} />
        <OcrTable label="Set code" field={result.ocr.set_code} />

        <DetectionTable detection={result.detection} />
        <TimingTable timing={result.timing_ms} />

        {debugEntries.length > 0 && (
          <table className="kv">
            <caption>Debug images (server paths)</caption>
            <tbody>
              {debugEntries.map(([key, path]) => (
                <tr key={key}>
                  <th scope="row">{key}</th>
                  <td>{path}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        <h3>Raw response</h3>
        <pre className="json">{json}</pre>
      </div>
    </details>
  );
}
