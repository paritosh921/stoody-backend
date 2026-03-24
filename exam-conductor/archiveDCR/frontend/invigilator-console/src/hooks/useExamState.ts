import { useCallback, useEffect, useRef, useState } from "react";
import type {
  SessionSummary,
  PenSyncRow,
  DongleRow,
  UploadProgressPayload,
  Alert,
  WebSocketEnvelope,
} from "@/types/api";
import { useWebSocket } from "./useWebSocket";

export interface ExamState {
  session: SessionSummary | null;
  pens: PenSyncRow[];
  dongles: DongleRow[];
  uploadProgress: UploadProgressPayload | null;
  alerts: Alert[];
}

/**
 * Aggregates WebSocket events into a single exam state object.
 * Generates alerts for pen timeouts, dongle failures, and sync errors.
 *
 * @param token  JWT auth token for the WebSocket connection
 * @param examId Exam session ID to subscribe to for live updates
 */
export function useExamState(token: string | null, examId: string | null) {
  const { connectionState, lastMessage } = useWebSocket(token, examId);
  const [state, setState] = useState<ExamState>({
    session: null,
    pens: [],
    dongles: [],
    uploadProgress: null,
    alerts: [],
  });

  const alertIdCounter = useRef(0);

  const addAlert = useCallback((severity: Alert["severity"], message: string) => {
    const id = `alert-${++alertIdCounter.current}`;
    setState((prev) => ({
      ...prev,
      alerts: [{ id, severity, message, timestamp: Date.now() }, ...prev.alerts].slice(0, 20),
    }));
  }, []);

  const dismissAlert = useCallback((alertId: string) => {
    setState((prev) => ({
      ...prev,
      alerts: prev.alerts.filter((a) => a.id !== alertId),
    }));
  }, []);

  useEffect(() => {
    if (!lastMessage) return;

    try {
      const envelope = JSON.parse(lastMessage.data as string) as WebSocketEnvelope;
      handleEvent(envelope);
    } catch {
      // Ignore malformed messages
    }
  }, [lastMessage]); // eslint-disable-line react-hooks/exhaustive-deps

  function handleEvent(envelope: WebSocketEnvelope) {
    switch (envelope.event_type) {
      case "session.snapshot":
        setState((prev) => ({
          ...prev,
          session: envelope.payload as SessionSummary,
        }));
        break;

      case "sync.progress":
        handleSyncProgress(envelope.payload as PenSyncRow);
        break;

      case "dongle.health":
        handleDongleHealth(envelope.payload as DongleRow);
        break;

      case "upload.progress":
        setState((prev) => ({
          ...prev,
          uploadProgress: envelope.payload as UploadProgressPayload,
        }));
        break;
    }
  }

  function handleSyncProgress(row: PenSyncRow) {
    setState((prev) => {
      const idx = prev.pens.findIndex((p) => p.pen_mac === row.pen_mac);
      const updated = idx >= 0
        ? prev.pens.map((p, i) => (i === idx ? row : p))
        : [...prev.pens, row];
      return { ...prev, pens: updated };
    });

    if (row.sync_status === "timeout") {
      addAlert("warning", `Pen ${row.pen_mac} timed out`);
    } else if (row.sync_status === "failed") {
      addAlert("error", `Pen ${row.pen_mac} sync failed`);
    }
  }

  function handleDongleHealth(row: DongleRow) {
    setState((prev) => {
      const idx = prev.dongles.findIndex((d) => d.dongle_mac === row.dongle_mac);
      const updated = idx >= 0
        ? prev.dongles.map((d, i) => (i === idx ? row : d))
        : [...prev.dongles, row];
      return { ...prev, dongles: updated };
    });

    if (row.status === "failed") {
      addAlert("error", `Dongle ${row.dongle_mac} failed`);
    } else if (row.status === "degraded") {
      addAlert("warning", `Dongle ${row.dongle_mac} degraded`);
    }
  }

  return { ...state, connectionState, dismissAlert };
}
