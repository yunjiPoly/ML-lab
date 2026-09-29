import { useCallback, useEffect, useState, type SyntheticEvent } from "react";
import { useCamera } from "../hooks/useCamera";
import CardGuideOverlay from "./CardGuideOverlay";
import Spinner from "./Spinner";

const JPEG_QUALITY = 0.92;

interface Captured {
  blob: Blob;
  url: string;
}

interface CameraCaptureProps {
  /** Disables every control (used while a recognition request is running). */
  disabled: boolean;
  /** Called with the JPEG blob when the user confirms the captured frame. */
  onSubmit: (blob: Blob) => void;
}

function canvasToBlob(canvas: HTMLCanvasElement, type: string, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Could not encode the captured frame."))),
      type,
      quality,
    );
  });
}

/**
 * Live camera preview with a card guide. Capture draws the current frame at the
 * video's native resolution and exports a JPEG; the user can retake or submit it.
 */
export default function CameraCapture({ disabled, onSubmit }: CameraCaptureProps) {
  const camera = useCamera();
  const [captured, setCaptured] = useState<Captured | null>(null);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [aspectRatio, setAspectRatio] = useState("4 / 3");

  // Release the preview object URL whenever it is replaced or the component unmounts.
  useEffect(
    () => () => {
      if (captured) {
        URL.revokeObjectURL(captured.url);
      }
    },
    [captured],
  );

  const handleMetadata = (event: SyntheticEvent<HTMLVideoElement>) => {
    const { videoWidth, videoHeight } = event.currentTarget;
    if (videoWidth > 0 && videoHeight > 0) {
      setAspectRatio(`${videoWidth} / ${videoHeight}`);
    }
  };

  const capture = useCallback(async () => {
    const video = camera.videoRef.current;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.videoWidth === 0) {
      setCaptureError("The camera is not ready yet. Please try again in a moment.");
      return;
    }
    try {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      if (!context) {
        throw new Error("Canvas 2D context is unavailable in this browser.");
      }
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await canvasToBlob(canvas, "image/jpeg", JPEG_QUALITY);
      setCaptureError(null);
      setCaptured({ blob, url: URL.createObjectURL(blob) });
    } catch (err) {
      setCaptureError(err instanceof Error ? err.message : String(err));
    }
  }, [camera.videoRef]);

  const retake = useCallback(() => {
    setCaptured(null);
    setCaptureError(null);
    if (camera.state !== "live") {
      void camera.start();
    }
  }, [camera]);

  const isLive = camera.state === "live";
  const showVideo = isLive && !captured;

  return (
    <div className="stack">
      <div className="stage" style={{ aspectRatio }}>
        <video
          ref={camera.videoRef}
          className={showVideo ? "stage__media" : "stage__media is-hidden"}
          autoPlay
          playsInline
          muted
          onLoadedMetadata={handleMetadata}
        />
        {captured && <img className="stage__media" src={captured.url} alt="Captured photo of the card" />}
        {showVideo && <CardGuideOverlay />}
        {!captured && camera.state === "idle" && (
          <div className="stage__placeholder">
            <p>The camera is off.</p>
            <p>Start it, place the card inside the guide and capture.</p>
          </div>
        )}
        {!captured && camera.state === "starting" && (
          <div className="stage__placeholder" role="status">
            <Spinner />
            <p>Starting camera...</p>
          </div>
        )}
        {!captured && camera.state === "error" && (
          <div className="stage__placeholder">
            <p>The camera could not be started.</p>
          </div>
        )}
      </div>

      {camera.error && (
        <p className="note note--error" role="alert">
          {camera.error}
        </p>
      )}
      {captureError && (
        <p className="note note--error" role="alert">
          {captureError}
        </p>
      )}

      <div className="actions">
        {captured ? (
          <>
            <button type="button" className="btn" onClick={retake} disabled={disabled}>
              Retake
            </button>
            <button
              type="button"
              className="btn btn--primary"
              onClick={() => onSubmit(captured.blob)}
              disabled={disabled}
            >
              Use photo
            </button>
          </>
        ) : isLive ? (
          <>
            <button type="button" className="btn" onClick={camera.stop} disabled={disabled}>
              Stop camera
            </button>
            <button type="button" className="btn btn--primary" onClick={() => void capture()} disabled={disabled}>
              Capture
            </button>
          </>
        ) : (
          <button
            type="button"
            className="btn btn--primary btn--block"
            onClick={() => void camera.start()}
            disabled={disabled || camera.state === "starting"}
          >
            {camera.state === "starting" ? "Starting camera..." : "Start camera"}
          </button>
        )}
      </div>
    </div>
  );
}
