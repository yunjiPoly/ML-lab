/**
 * Camera access hook built on `navigator.mediaDevices.getUserMedia`.
 *
 * Owns the `MediaStream`, attaches it to the `<video>` element referenced by
 * `videoRef`, and stops every track on `stop()` and on unmount so the camera
 * light goes off as soon as the scanner is left.
 */
import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

export type CameraState = "idle" | "starting" | "live" | "error";

export interface UseCamera {
  /** Attach to the `<video>` element that displays the live preview. */
  videoRef: RefObject<HTMLVideoElement | null>;
  state: CameraState;
  /** Friendly error message when `state === "error"`. */
  error: string | null;
  start: () => Promise<void>;
  stop: () => void;
}

/** Preferred constraints: rear camera, Full HD when the device offers it. */
export const CAMERA_CONSTRAINTS: MediaStreamConstraints = {
  audio: false,
  video: {
    facingMode: { ideal: "environment" },
    width: { ideal: 1920 },
    height: { ideal: 1080 },
  },
};

const UPLOAD_HINT = 'You can still use the "Upload image" option.';

/** Returns a message when the camera cannot work in this page context, otherwise `null`. */
export function cameraUnavailableReason(): string | null {
  if (typeof window === "undefined" || typeof navigator === "undefined") {
    return "The camera is not available in this environment.";
  }
  if (!window.isSecureContext) {
    return (
      "The camera needs a secure context (HTTPS or localhost). Open this page over HTTPS " +
      `(start the dev server with VITE_HTTPS=true) or use the upload option. ${UPLOAD_HINT}`
    );
  }
  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
    return `This browser does not support camera access. ${UPLOAD_HINT}`;
  }
  return null;
}

/** Maps `getUserMedia` errors to friendly, actionable messages. */
export function describeCameraError(error: unknown): string {
  const name = error instanceof Error ? error.name : "";
  const message = error instanceof Error ? error.message : String(error ?? "");
  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return `Camera permission was denied. Allow camera access for this site in the browser settings, or use the upload option. ${UPLOAD_HINT}`;
    case "NotFoundError":
    case "DevicesNotFoundError":
      return `No camera was found on this device. ${UPLOAD_HINT}`;
    case "NotReadableError":
    case "TrackStartError":
      return `The camera could not be started; it may be in use by another app. ${UPLOAD_HINT}`;
    case "OverconstrainedError":
      return `The camera does not support the requested settings. ${UPLOAD_HINT}`;
    case "SecurityError":
      return `Camera access is blocked in this context (use HTTPS or localhost). ${UPLOAD_HINT}`;
    default: {
      const detail = [name, message].filter(Boolean).join(": ");
      return `Could not start the camera${detail ? ` (${detail})` : ""}. ${UPLOAD_HINT}`;
    }
  }
}

function stopStream(stream: MediaStream | null): void {
  stream?.getTracks().forEach((track) => track.stop());
}

/** Requests the preferred stream, relaxing the constraints once if the device rejects them. */
async function acquireStream(): Promise<MediaStream> {
  try {
    return await navigator.mediaDevices.getUserMedia(CAMERA_CONSTRAINTS);
  } catch (error) {
    if (error instanceof Error && error.name === "OverconstrainedError") {
      return navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: "environment" } });
    }
    throw error;
  }
}

export function useCamera(): UseCamera {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mountedRef = useRef(true);
  const [state, setState] = useState<CameraState>("idle");
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    stopStream(streamRef.current);
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setState("idle");
  }, []);

  const start = useCallback(async () => {
    const unavailable = cameraUnavailableReason();
    if (unavailable) {
      setError(unavailable);
      setState("error");
      return;
    }
    stopStream(streamRef.current);
    streamRef.current = null;
    setError(null);
    setState("starting");
    try {
      const stream = await acquireStream();
      if (!mountedRef.current) {
        stopStream(stream);
        return;
      }
      const video = videoRef.current;
      if (!video) {
        stopStream(stream);
        throw new Error("The video element is not mounted.");
      }
      streamRef.current = stream;
      video.srcObject = stream;
      stream.getVideoTracks()[0]?.addEventListener("ended", () => {
        if (streamRef.current === stream) {
          stop();
        }
      });
      try {
        await video.play();
      } catch {
        // Muted inline playback normally starts by itself; a rejected play()
        // (e.g. interrupted by a re-render) does not invalidate the stream.
      }
      if (mountedRef.current) {
        setState("live");
      }
    } catch (err) {
      if (!mountedRef.current) {
        return;
      }
      setError(describeCameraError(err));
      setState("error");
    }
  }, [stop]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      stopStream(streamRef.current);
      streamRef.current = null;
    };
  }, []);

  return { videoRef, state, error, start, stop };
}
