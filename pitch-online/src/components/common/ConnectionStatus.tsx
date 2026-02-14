import { useAppState } from "../../contexts/AppContext";

export default function ConnectionStatus() {
  const { connectionStatus } = useAppState();

  if (connectionStatus === "connected") return null;

  const isWarning = connectionStatus === "connecting";
  const text = isWarning ? "Connecting..." : "Disconnected";

  return (
    <div className={`connection-badge ${isWarning ? "connection-badge--warning" : "connection-badge--error"}`}>
      {text}
    </div>
  );
}
