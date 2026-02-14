import { useAppState } from "../../contexts/AppContext";

export default function ConnectionStatus() {
  const { connectionStatus } = useAppState();

  if (connectionStatus === "connected") return null;

  const color = connectionStatus === "connecting" ? "#ffc107" : "#dc3545";
  const text = connectionStatus === "connecting" ? "Connecting..." : "Disconnected";

  return (
    <div
      style={{
        position: "fixed",
        top: 10,
        right: 10,
        zIndex: 9999,
        padding: "4px 12px",
        borderRadius: 4,
        backgroundColor: color,
        color: "white",
        fontSize: "0.75rem",
      }}
    >
      {text}
    </div>
  );
}
