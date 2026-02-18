import type { SeatInfo } from "../../types";

interface SeatSlotProps {
  seat: SeatInfo | null;
  isLocal: boolean;
  teamLabel: string;
  position: string;
}

export default function SeatSlot({ seat, isLocal, teamLabel, position }: SeatSlotProps) {
  return (
    <div
      className="d-flex flex-column align-items-center"
      style={{ minWidth: 100 }}
    >
      <small className="lobby-muted mb-1">{position} - {teamLabel}</small>
      <div
        className={`rounded p-2 text-center seat-slot${isLocal ? " seat-slot--local" : ""}`}
      >
        {seat ? (
          <>
            <div className={isLocal ? "seat-slot-name--local" : ""}>
              {seat.displayName}
              {isLocal && " (You)"}
            </div>
            {seat.isAI && <small className="text-warning">AI</small>}
            {!seat.isAI && !seat.isConnected && <small className="text-danger">Disconnected</small>}
          </>
        ) : (
          <div className="lobby-muted">Waiting...</div>
        )}
      </div>
    </div>
  );
}
