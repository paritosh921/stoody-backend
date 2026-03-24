export type ConnectionState = "connecting" | "connected" | "disconnected" | "error";

export interface UseWebSocketReturn {
  connectionState: ConnectionState;
  lastMessage: MessageEvent | null;
  sendMessage: (data: string) => void;
}
