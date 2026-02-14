import { useRef, useEffect, useState } from "react";
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

function useScoreBump(value: number): boolean {
  const prev = useRef(value);
  const [bumping, setBumping] = useState(false);

  useEffect(() => {
    if (prev.current !== value && prev.current !== 0) {
      setBumping(true);
      const t = setTimeout(() => setBumping(false), 400);
      prev.current = value;
      return () => clearTimeout(t);
    }
    prev.current = value;
  }, [value]);

  return bumping;
}

export default function Scoreboard({ scores, roundScores, currentBid, currentHighBidder, seats, localSeat }: ScoreboardProps) {
  const localTeam = localSeat % 2;
  const otherTeam = 1 - localTeam;
  const localBump = useScoreBump(scores[localTeam]);
  const otherBump = useScoreBump(scores[otherTeam]);

  const bidderName = seats.find((s) => s.seatIndex === currentHighBidder)?.displayName ?? "";

  return (
    <div className="scoreboard" style={{ color: "white", fontSize: "0.85rem" }}>
      <div style={{ fontWeight: "bold", color: "#7cfc00" }}>
        Team {localTeam === 0 ? "A" : "B"}:{" "}
        <span className={localBump ? "anim-score-bump" : ""} style={{ display: "inline-block" }}>
          {scores[localTeam]}
        </span>{" "}
        (Round: {roundScores[localTeam]})
      </div>
      <div>
        Team {otherTeam === 0 ? "A" : "B"}:{" "}
        <span className={otherBump ? "anim-score-bump" : ""} style={{ display: "inline-block" }}>
          {scores[otherTeam]}
        </span>{" "}
        (Round: {roundScores[otherTeam]})
      </div>
      {currentBid > 0 && (
        <div>
          Bid: {bidDisplay(currentBid)} by {bidderName}
        </div>
      )}
    </div>
  );
}
