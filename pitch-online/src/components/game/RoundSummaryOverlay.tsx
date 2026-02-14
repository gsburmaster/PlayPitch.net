import type { RoundEndData } from "../../contexts/GameContext";
import { bidDisplay } from "../../types";

interface RoundSummaryOverlayProps {
  data: RoundEndData;
}

export default function RoundSummaryOverlay({ data }: RoundSummaryOverlayProps) {
  const bidderTeamName = data.bidderTeam === 0 ? "Team A" : "Team B";
  const otherTeamName = data.bidderTeam === 0 ? "Team B" : "Team A";
  const otherTeam = 1 - data.bidderTeam;

  return (
    <div className="game-overlay overlay-anim">
      <div className="overlay-content overlay-content-anim text-center">
        <h3>Round Over!</h3>
        <p>
          {bidderTeamName} took {data.roundScores[data.bidderTeam]} points (Bid: {bidDisplay(data.bidAmount)})
          {data.bidMade ? (
            <span className="text-success"> -- Made it!</span>
          ) : (
            <span className="text-danger">
              {" "}-- Set! {data.scoreDeltas[data.bidderTeam] > 0 ? "+" : ""}
              {data.scoreDeltas[data.bidderTeam]}
            </span>
          )}
        </p>
        <p>
          {otherTeamName} took {data.roundScores[otherTeam]} points
        </p>
        <p style={{ fontWeight: "bold" }}>
          Scores: Team A {data.totalScores[0]} | Team B {data.totalScores[1]}
        </p>
      </div>
    </div>
  );
}
