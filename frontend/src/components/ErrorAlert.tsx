interface ErrorAlertProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  onDismiss?: () => void;
}

/** Inline error banner with optional Retry / Dismiss actions. */
export default function ErrorAlert({ title = "Request failed", message, onRetry, onDismiss }: ErrorAlertProps) {
  return (
    <div className="alert alert--error" role="alert">
      <div className="alert__body">
        <div className="alert__title">{title}</div>
        <div className="alert__msg">{message}</div>
      </div>
      {(onRetry || onDismiss) && (
        <div className="alert__actions">
          {onRetry && (
            <button type="button" className="btn btn--small btn--primary" onClick={onRetry}>
              Retry
            </button>
          )}
          {onDismiss && (
            <button type="button" className="btn btn--small" onClick={onDismiss}>
              Dismiss
            </button>
          )}
        </div>
      )}
    </div>
  );
}
