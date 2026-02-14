import { useAppState } from "../../contexts/AppContext";

interface Props {
  onReconnect: () => void;
}

export default function DisconnectedOverlay({ onReconnect }: Props) {
  const { connectionStatus } = useAppState();

  if (connectionStatus === "connected") return null;

  const isConnecting = connectionStatus === "connecting";

  return (
    <div style={{
      position: "fixed",
      inset: 0,
      zIndex: 9998,
      backgroundColor: "rgba(0,0,0,0.6)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    }}>
      <div style={{
        backgroundColor: "#1a1a2e",
        border: "1px solid rgba(255,255,255,0.15)",
        borderRadius: 12,
        padding: "32px 40px",
        textAlign: "center",
        color: "white",
        maxWidth: 380,
      }}>
        <div style={{ fontSize: "2rem", marginBottom: 12 }}>
          {isConnecting ? "\u26A1" : "\u26A0\uFE0F"}
        </div>
        <h4 style={{ marginBottom: 8 }}>
          {isConnecting ? "Reconnecting..." : "Disconnected"}
        </h4>
        <p style={{ color: "#aaa", fontSize: "0.9rem", marginBottom: 20 }}>
          {isConnecting
            ? "Attempting to reconnect to the game server..."
            : "Lost connection to the game server. Your seat is reserved for 60 seconds."}
        </p>
        {!isConnecting && (
          <button
            onClick={onReconnect}
            style={{
              padding: "10px 28px",
              borderRadius: 6,
              border: "none",
              backgroundColor: "#28a745",
              color: "white",
              fontSize: "1rem",
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            Reconnect Now
          </button>
        )}
        {isConnecting && (
          <div style={{
            width: 32,
            height: 32,
            border: "3px solid rgba(255,255,255,0.2)",
            borderTopColor: "#ffc107",
            borderRadius: "50%",
            animation: "spin 0.8s linear infinite",
            margin: "0 auto",
          }} />
        )}
        <style>{`
          @keyframes spin {
            to { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    </div>
  );
}
