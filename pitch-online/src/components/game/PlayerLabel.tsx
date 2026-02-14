interface PlayerLabelProps {
  name: string;
  isDealer: boolean;
  isConnected: boolean;
  isLocal: boolean;
  isCurrentPlayer: boolean;
}

export default function PlayerLabel({ name, isDealer, isConnected, isLocal, isCurrentPlayer }: PlayerLabelProps) {
  return (
    <div className="player-label d-flex align-items-center gap-1 justify-content-center">
      <span
        className={`connection-dot ${isConnected ? "bg-success" : "bg-danger"}`}
        style={{ width: 8, height: 8, borderRadius: "50%", display: "inline-block" }}
      />
      <span
        className={isCurrentPlayer ? "anim-turn-glow" : ""}
        style={{
          fontWeight: isCurrentPlayer ? "bold" : "normal",
          color: isLocal ? "#7cfc00" : "white",
        }}
      >
        {name}
        {isLocal && " (You)"}
      </span>
      {isDealer && (
        <span className="badge bg-warning text-dark" style={{ fontSize: "0.65rem" }}>
          D
        </span>
      )}
    </div>
  );
}
