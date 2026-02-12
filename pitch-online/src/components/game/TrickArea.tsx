import type { SeatIndex, TrickCard } from "../../types";
import Card from "./Card";
import "../../styles/trick-label.css";

interface TrickAreaProps {
  trick: TrickCard[];
  localSeat: SeatIndex;
  winnerSeat?: number | null;
  seatNames: Record<number, string>;
}

// Map absolute seat to position relative to local player
function getRelativePosition(seatIndex: SeatIndex, localSeat: SeatIndex): "bottom" | "left" | "top" | "right" {
  const rel = ((seatIndex - localSeat + 4) % 4) as 0 | 1 | 2 | 3;
  return (["bottom", "left", "top", "right"] as const)[rel];
}

const POSITION_STYLES: Record<string, React.CSSProperties> = {
  bottom: { bottom: 0, left: "50%", transform: "translateX(-50%)" },
  top: { top: 0, left: "50%", transform: "translateX(-50%)" },
  left: { left: 0, top: "50%", transform: "translateY(-50%)" },
  right: { right: 0, top: "50%", transform: "translateY(-50%)" },
};

export default function TrickArea({ trick, localSeat, winnerSeat, seatNames }: TrickAreaProps) {
  return (
    <div className="trick-area" style={{ position: "relative", width: 200, height: 160, margin: "0 auto" }}>
      {trick.map((entry, idx) => {
        const pos = getRelativePosition(entry.seatIndex, localSeat);
        const isWinner = winnerSeat === entry.seatIndex;
        const isLocal = entry.seatIndex === localSeat;
        const name = isLocal ? "You" : (seatNames[entry.seatIndex] ?? "");

        return (
          <div key={idx} className="trick-card-slot" style={{ position: "absolute", ...POSITION_STYLES[pos] }}>
            <Card card={entry.card} faceUp highlighted={isWinner} />
            <span className={`trick-label${isLocal ? " trick-label-local" : ""}`}>
              {name}
            </span>
          </div>
        );
      })}
    </div>
  );
}
