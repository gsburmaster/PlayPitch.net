import { bidDisplay } from "../../types";
import type { SeatIndex, SeatInfo } from "../../types";

interface ScoreboardProps {
  scores: [number, number];
  roundScores: [number, number];
  currentBid: number;
  currentHighBidder: number;
  seats: SeatInfo[];
  localSeat: SeatIndex;
}

export default function Scoreboard({ scores, roundScores, currentBid, currentHighBidder, seats, localSeat }: ScoreboardProps) {
  const localTeam = localSeat % 2;
  const otherTeam = 1 - localTeam;

  const bidderName = seats.find((s) => s.seatIndex === currentHighBidder)?.displayName ?? "";

  return (
    <div className="scoreboard" style={{ color: "white", fontSize: "0.85rem" }}>
      <div style={{ fontWeight: "bold", color: "#7cfc00" }}>
        Team {localTeam === 0 ? "A" : "B"}: {scores[localTeam]} (Round: {roundScores[localTeam]})
      </div>
      <div>
        Team {otherTeam === 0 ? "A" : "B"}: {scores[otherTeam]} (Round: {roundScores[otherTeam]})
      </div>
      {currentBid > 0 && (
        <div>
          Bid: {bidDisplay(currentBid)} by {bidderName}
        </div>
      )}
    </div>
  );
}
