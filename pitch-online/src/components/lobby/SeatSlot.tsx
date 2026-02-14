import type { SeatInfo } from "../../types";

interface SeatSlotProps {
  seat: SeatInfo | null;
  isLocal: boolean;
  teamLabel: string;
  position: string;
}

export default function SeatSlot({ seat, isLocal, teamLabel, position }: SeatSlotProps) {
  const borderColor = isLocal ? "#28a745" : "rgba(255,255,255,0.3)";

  return (
    <div
      className="d-flex flex-column align-items-center"
      style={{ minWidth: 100 }}
    >
      <small className="text-muted mb-1">{position} - {teamLabel}</small>
      <div
        className="rounded p-2 text-center"
        style={{
          border: `2px solid ${borderColor}`,
          backgroundColor: "rgba(0,0,0,0.3)",
          minWidth: 120,
          color: "white",
        }}
      >
        {seat ? (
          <>
            <div style={{ fontWeight: isLocal ? "bold" : "normal" }}>
              {seat.displayName}
              {isLocal && " (You)"}
            </div>
            {seat.isAI && <small className="text-warning">AI</small>}
            {!seat.isAI && !seat.isConnected && <small className="text-danger">Disconnected</small>}
          </>
        ) : (
          <div className="text-muted">Waiting...</div>
        )}
      </div>
    </div>
  );
}
