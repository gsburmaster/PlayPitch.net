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
      />
      <span
        className={`player-name${isLocal ? " player-name--local" : ""}${isCurrentPlayer ? " player-name--active anim-turn-glow" : ""}`}
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
