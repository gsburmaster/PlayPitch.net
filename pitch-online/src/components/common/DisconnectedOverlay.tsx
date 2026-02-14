import { useAppState } from "../../contexts/AppContext";

interface Props {
  onReconnect: () => void;
}

export default function DisconnectedOverlay({ onReconnect }: Props) {
  const { connectionStatus } = useAppState();

  if (connectionStatus === "connected") return null;

  const isConnecting = connectionStatus === "connecting";

  return (
    <div className="disconnected-overlay">
      <div className="disconnected-box">
        <div className="disconnected-icon">
          {isConnecting ? "\u26A1" : "\u26A0\uFE0F"}
        </div>
        <h4 style={{ marginBottom: 8 }}>
          {isConnecting ? "Reconnecting..." : "Disconnected"}
        </h4>
        <p className="disconnected-text">
          {isConnecting
            ? "Attempting to reconnect to the game server..."
            : "Lost connection to the game server. Your seat is reserved for 60 seconds."}
        </p>
        {!isConnecting && (
          <button onClick={onReconnect} className="disconnected-btn">
            Reconnect Now
          </button>
        )}
        {isConnecting && (
          <div className="spinner" />
        )}
      </div>
    </div>
  );
}
