// One connection to the game server: opens the socket, authenticates, pings,
// and reconnects with backoff when the connection drops, but never after we
// close it ourselves.

const PING_INTERVAL_MS = 25000;
const MAX_BACKOFF_MS = 16000;

export interface ConnectionOptions {
  url: string;
  authMessage: Record<string, unknown>;
  onMessage: (msg: Record<string, unknown>) => void;
  /** "connected" is the caller's call: it comes from the server's auth reply. */
  onStatus: (status: "connecting" | "disconnected") => void;
  createSocket?: (url: string) => WebSocket;
}

export interface Connection {
  send: (msg: Record<string, unknown>) => void;
  reconnectNow: () => void;
  close: () => void;
}

export function openConnection({
  url,
  authMessage,
  onMessage,
  onStatus,
  createSocket = (u) => new WebSocket(u),
}: ConnectionOptions): Connection {
  let ws: WebSocket | null = null;
  let attempt = 0;
  let pingTimer: ReturnType<typeof setInterval> | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  function stopTimers() {
    if (pingTimer) clearInterval(pingTimer);
    if (reconnectTimer) clearTimeout(reconnectTimer);
    pingTimer = null;
    reconnectTimer = null;
  }

  function dropSocket() {
    if (!ws) return;
    // Detach first, so a socket we close ourselves never looks like a dropped connection.
    ws.onopen = null;
    ws.onmessage = null;
    ws.onclose = null;
    ws.close();
    ws = null;
  }

  function send(msg: Record<string, unknown>) {
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  }

  function connect() {
    stopTimers();
    dropSocket();
    onStatus("connecting");
    const socket = createSocket(url);
    ws = socket;
    socket.onopen = () => {
      attempt = 0;
      socket.send(JSON.stringify(authMessage));
      pingTimer = setInterval(() => send({ type: "ping" }), PING_INTERVAL_MS);
    };
    socket.onmessage = (event) => onMessage(JSON.parse(event.data));
    socket.onclose = () => {
      ws = null;
      stopTimers();
      onStatus("disconnected");
      const delay = Math.min(1000 * 2 ** attempt, MAX_BACKOFF_MS);
      attempt++;
      reconnectTimer = setTimeout(connect, delay);
    };
  }

  connect();

  return {
    send,
    reconnectNow: () => {
      attempt = 0;
      connect();
    },
    close: () => {
      stopTimers();
      dropSocket();
    },
  };
}
