import { useCallback, useEffect, useRef, useState } from "react";
import type { ConnectionState, UseWebSocketReturn } from "@/types/ws";

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;
const RECONNECT_BACKOFF = 2;

/**
 * WebSocket hook for the invigilator console.
 * Connects to ws://host/api/v1/invigilator/ws?token=...
 * After connection opens, sends a subscribe frame for the given examId.
 * Handles auto-reconnect with exponential backoff.
 *
 * Backend handshake contract:
 *   1. Connect with `?token=<JWT>` query param
 *   2. Send `{"type":"subscribe","exam_id":"..."}` after open
 *   3. Receive `{"event_type":"subscribed",...}` confirmation
 *   4. Receive 1 Hz `session.snapshot` pushes
 */
export function useWebSocket(
  token: string | null,
  examId: string | null,
): UseWebSocketReturn {
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected");
  const [lastMessage, setLastMessage] = useState<MessageEvent | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelay = useRef(RECONNECT_BASE_MS);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const intentionalClose = useRef(false);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimer.current !== null) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (!token || !examId) return;

    clearReconnectTimer();
    intentionalClose.current = false;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/api/v1/invigilator/ws?token=${encodeURIComponent(token)}`;

    setConnectionState("connecting");
    const ws = new WebSocket(url);

    ws.onopen = () => {
      // Send the subscribe frame immediately after connection opens.
      // The backend waits for this before starting snapshot pushes.
      ws.send(JSON.stringify({ type: "subscribe", exam_id: examId }));
      setConnectionState("connected");
      reconnectDelay.current = RECONNECT_BASE_MS;
    };

    ws.onmessage = (event: MessageEvent) => {
      setLastMessage(event);
    };

    ws.onerror = () => {
      setConnectionState("error");
    };

    ws.onclose = () => {
      wsRef.current = null;
      if (intentionalClose.current) {
        setConnectionState("disconnected");
        return;
      }
      setConnectionState("disconnected");
      scheduleReconnect();
    };

    wsRef.current = ws;
  }, [token, examId, clearReconnectTimer]);

  const scheduleReconnect = useCallback(() => {
    const delay = reconnectDelay.current;
    reconnectDelay.current = Math.min(delay * RECONNECT_BACKOFF, RECONNECT_MAX_MS);
    reconnectTimer.current = setTimeout(() => {
      connect();
    }, delay);
  }, [connect]);

  const sendMessage = useCallback((data: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      intentionalClose.current = true;
      clearReconnectTimer();
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect, clearReconnectTimer]);

  return { connectionState, lastMessage, sendMessage };
}
