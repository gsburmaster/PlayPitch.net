import { Button } from "react-bootstrap";
import type { GameOverData } from "../../contexts/GameContext";
import type { SeatIndex } from "../../types";

interface GameOverOverlayProps {
  data: GameOverData;
  localSeat: SeatIndex;
  onPlayAgain: () => void;
  onLeave: () => void;
}

export default function GameOverOverlay({ data, localSeat, onPlayAgain, onLeave }: GameOverOverlayProps) {
  const localTeam = localSeat % 2;
  const didWin = data.winner === localTeam;

  return (
    <div className="game-overlay overlay-anim">
      <div className="overlay-content overlay-content-anim text-center">
        <h2 className={didWin ? "game-over-win" : "game-over-lose"}>
          {didWin ? "You Win!" : "You Lose"}
        </h2>
        <h4>
          Team A: {data.finalScores[0]} -- Team B: {data.finalScores[1]}
        </h4>
        <p>
          {data.winner === 0 ? "Team A" : "Team B"} wins!
        </p>
        <div className="d-flex gap-2 justify-content-center mt-3">
          <Button variant="success" onClick={onPlayAgain}>
            Play Again
          </Button>
          <Button variant="outline-light" onClick={onLeave}>
            Leave
          </Button>
        </div>
      </div>
    </div>
  );
}
