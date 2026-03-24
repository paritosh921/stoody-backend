import { useEffect, useState } from "react";

interface TimerDisplayProps {
  /** Remaining seconds as last reported by the server. */
  remainingSeconds: number;
}

/**
 * Large countdown timer display.
 * Locally ticks every second between server updates so the display
 * stays smooth even if WebSocket messages arrive at a slower cadence.
 */
export function TimerDisplay({ remainingSeconds }: TimerDisplayProps) {
  const [localRemaining, setLocalRemaining] = useState(remainingSeconds);

  // Re-sync whenever the server pushes a new value
  useEffect(() => {
    setLocalRemaining(remainingSeconds);
  }, [remainingSeconds]);

  // Local tick
  useEffect(() => {
    if (localRemaining <= 0) return;
    const id = setInterval(() => {
      setLocalRemaining((prev) => Math.max(0, prev - 1));
    }, 1000);
    return () => clearInterval(id);
  }, [localRemaining > 0]); // eslint-disable-line react-hooks/exhaustive-deps

  const hours = Math.floor(localRemaining / 3600);
  const minutes = Math.floor((localRemaining % 3600) / 60);
  const seconds = localRemaining % 60;

  const pad = (n: number) => String(n).padStart(2, "0");
  const isUrgent = localRemaining <= 300; // 5 minutes

  return (
    <div
      className={`flex items-center justify-center rounded-xl px-8 py-4 font-mono text-5xl tabular-nums tracking-wider ${
        isUrgent
          ? "bg-red-950/60 text-red-400 animate-pulse"
          : "bg-gray-900 text-white"
      }`}
    >
      {hours > 0 && <span>{pad(hours)}:</span>}
      <span>{pad(minutes)}</span>
      <span className="mx-1">:</span>
      <span>{pad(seconds)}</span>
    </div>
  );
}
