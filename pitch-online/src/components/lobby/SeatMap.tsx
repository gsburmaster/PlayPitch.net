import type { SeatIndex, SeatInfo } from "../../types";
import SeatSlot from "./SeatSlot";

interface SeatMapProps {
  seats: SeatInfo[];
  localSeat: SeatIndex | null;
}

const POSITIONS = ["Bottom", "Left", "Top", "Right"];
const TEAM_LABELS = ["Team A", "Team B", "Team A", "Team B"];

export default function SeatMap({ seats, localSeat }: SeatMapProps) {
  const seatMap = new Map<number, SeatInfo>();
  for (const s of seats) seatMap.set(s.seatIndex, s);

  return (
    <div className="d-flex flex-wrap justify-content-center gap-3 my-3">
      {([0, 1, 2, 3] as SeatIndex[]).map((idx) => (
        <SeatSlot
          key={idx}
          seat={seatMap.get(idx) ?? null}
          isLocal={idx === localSeat}
          teamLabel={TEAM_LABELS[idx]}
          position={POSITIONS[idx]}
        />
      ))}
    </div>
  );
}
