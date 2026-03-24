import type { Alert } from "@/types/api";

interface AlertBannerProps {
  alerts: Alert[];
  onDismiss: (alertId: string) => void;
}

const SEVERITY_STYLES: Record<
  Alert["severity"],
  { bg: string; border: string; icon: string }
> = {
  error: {
    bg: "bg-red-950/60",
    border: "border-red-800/50",
    icon: "!",
  },
  warning: {
    bg: "bg-yellow-950/60",
    border: "border-yellow-800/50",
    icon: "!!",
  },
  info: {
    bg: "bg-blue-950/60",
    border: "border-blue-800/50",
    icon: "i",
  },
};

/**
 * Stacked alert banner displaying failure/warning/info messages.
 * Newest alerts appear at the top. Each alert can be dismissed.
 */
export function AlertBanner({ alerts, onDismiss }: AlertBannerProps) {
  if (alerts.length === 0) return null;

  return (
    <div className="flex flex-col gap-2">
      {alerts.map((alert) => {
        const style = SEVERITY_STYLES[alert.severity];
        const timeStr = new Date(alert.timestamp).toLocaleTimeString();

        return (
          <div
            key={alert.id}
            className={`flex items-start gap-3 rounded-lg border ${style.border} ${style.bg} px-4 py-3`}
          >
            {/* Severity icon */}
            <span className="mt-0.5 flex-shrink-0 w-5 h-5 rounded-full bg-gray-800 flex items-center justify-center text-[10px] font-bold text-gray-200">
              {style.icon}
            </span>

            {/* Message */}
            <div className="flex-1 min-w-0">
              <p className="text-sm text-gray-200 break-words">
                {alert.message}
              </p>
              <p className="text-[10px] text-gray-500 mt-0.5">{timeStr}</p>
            </div>

            {/* Dismiss button */}
            <button
              onClick={() => onDismiss(alert.id)}
              className="flex-shrink-0 text-gray-500 hover:text-gray-300 transition-colors text-lg leading-none mt-0.5"
              aria-label="Dismiss alert"
            >
              x
            </button>
          </div>
        );
      })}
    </div>
  );
}
