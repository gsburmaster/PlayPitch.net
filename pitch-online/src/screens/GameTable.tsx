import { useAppState } from "../contexts/AppContext";
import { useGameState } from "../contexts/GameContext";
import { useWebSocket } from "../hooks/useWebSocket";
import type { SeatIndex } from "../types";
import Scoreboard from "../components/game/Scoreboard";
import TrumpIndicator from "../components/game/TrumpIndicator";
import PhaseIndicator from "../components/game/PhaseIndicator";
import PlayerArea from "../components/game/PlayerArea";
import TrickArea from "../components/game/TrickArea";
import ActionPanel from "../components/game/ActionPanel";
import RoundSummaryOverlay from "../components/game/RoundSummaryOverlay";
import GameOverOverlay from "../components/game/GameOverOverlay";
import DisconnectedOverlay from "../components/common/DisconnectedOverlay";
import TrickHistoryDrawer from "../components/game/TrickHistoryDrawer";
import { bidDisplay } from "../types";
import "../styles/table.css";

export default function GameTable() {
  const { seatIndex: localSeat } = useAppState();
  const game = useGameState();
  const { sendAction, playAgain, leaveRoom, reconnect } = useWebSocket();

  if (localSeat === null) return null;

  const isMyTurn = game.currentPlayer === localSeat;

  // Map relative positions to absolute seats
  const seatOrder = [0, 1, 2, 3].map(
    (offset) => ((localSeat + offset) % 4) as SeatIndex
  );

  const getSeatInfo = (seat: SeatIndex) => game.seats.find((s) => s.seatIndex === seat);

  // Map seat index → display name for trick area labels
  const seatNames: Record<number, string> = {};
  for (const s of game.seats) seatNames[s.seatIndex] = s.displayName;

  // Announcement for choose trump phase
  const trumpChooserName = game.phase === 1
    ? getSeatInfo(game.currentHighBidder as SeatIndex)?.displayName
    : undefined;

  const centerAnnouncement = game.phase === 1 && !isMyTurn && trumpChooserName
    ? `${trumpChooserName} won the bid with ${bidDisplay(game.currentBid)}`
    : null;

  return (
    <div className="table-bg game-table">
      {/* Top bar */}
      <div className="game-topbar">
        <Scoreboard
          scores={game.scores}
          roundScores={game.roundScores}
          currentBid={game.currentBid}
          currentHighBidder={game.currentHighBidder}
          seats={game.seats}
          localSeat={localSeat}
        />
        <div className="d-flex align-items-center gap-2">
          <TrumpIndicator trumpSuit={game.trumpSuit} />
          <PhaseIndicator phase={game.phase} />
          <TrickHistoryDrawer
            trickHistory={game.trickHistory}
            localSeat={localSeat}
            seatNames={seatNames}
          />
        </div>
      </div>

      {/* Top player (partner) */}
      <div className="game-position-top">
        <PlayerArea
          seatIndex={seatOrder[2]}
          displayName={getSeatInfo(seatOrder[2])?.displayName ?? ""}
          isLocal={false}
          isDealer={game.dealer === seatOrder[2]}
          isConnected={getSeatInfo(seatOrder[2])?.isConnected ?? false}
          isCurrentPlayer={game.currentPlayer === seatOrder[2]}
          cards={[]}
          cardCount={game.playerCardCounts[seatOrder[2]]}
          bidEntry={game.bidHistory.find((b) => b.seatIndex === seatOrder[2])}
          position="top"
        />
      </div>

      {/* Left player (opponent) */}
      <div className="game-position-left">
        <PlayerArea
          seatIndex={seatOrder[1]}
          displayName={getSeatInfo(seatOrder[1])?.displayName ?? ""}
          isLocal={false}
          isDealer={game.dealer === seatOrder[1]}
          isConnected={getSeatInfo(seatOrder[1])?.isConnected ?? false}
          isCurrentPlayer={game.currentPlayer === seatOrder[1]}
          cards={[]}
          cardCount={game.playerCardCounts[seatOrder[1]]}
          bidEntry={game.bidHistory.find((b) => b.seatIndex === seatOrder[1])}
          position="left"
        />
      </div>

      {/* Right player (opponent) */}
      <div className="game-position-right">
        <PlayerArea
          seatIndex={seatOrder[3]}
          displayName={getSeatInfo(seatOrder[3])?.displayName ?? ""}
          isLocal={false}
          isDealer={game.dealer === seatOrder[3]}
          isConnected={getSeatInfo(seatOrder[3])?.isConnected ?? false}
          isCurrentPlayer={game.currentPlayer === seatOrder[3]}
          cards={[]}
          cardCount={game.playerCardCounts[seatOrder[3]]}
          bidEntry={game.bidHistory.find((b) => b.seatIndex === seatOrder[3])}
          position="right"
        />
      </div>

      {/* Center: trick area + announcements */}
      <div className="game-center">
        {centerAnnouncement && (
          <div className="text-white text-center mb-2 center-announcement">
            {centerAnnouncement}
          </div>
        )}
        <TrickArea
          trick={game.currentTrick}
          localSeat={localSeat}
          winnerSeat={game.lastTrickResult?.winner}
          seatNames={seatNames}
        />
        {game.lastTrickResult && (
          <div className="text-warning text-center mt-1 trick-result-text">
            {game.lastTrickResult.winnerName} wins the trick
            {game.lastTrickResult.pointsWon > 0 && ` (+${game.lastTrickResult.pointsWon} pts)`}
          </div>
        )}
      </div>

      {/* Bottom player (local) */}
      <div className="game-position-bottom">
        <PlayerArea
          seatIndex={localSeat}
          displayName={getSeatInfo(localSeat)?.displayName ?? "You"}
          isLocal
          isDealer={game.dealer === localSeat}
          isConnected
          isCurrentPlayer={isMyTurn}
          cards={game.myHand}
          cardCount={game.myHand.length}
          bidEntry={game.bidHistory.find((b) => b.seatIndex === localSeat)}
          actionMask={game.actionMask}
          isMyTurn={isMyTurn}
          phase={game.phase}
          onPlayCard={(idx) => sendAction(idx)}
          position="bottom"
        />

        {/* Action panel */}
        <ActionPanel
          phase={game.phase}
          actionMask={game.actionMask}
          isMyTurn={isMyTurn}
          currentBid={game.currentBid}
          currentHighBidder={game.currentHighBidder}
          currentPlayer={game.currentPlayer}
          seats={game.seats}
          trumpChooserName={trumpChooserName}
          onAction={sendAction}
        />
      </div>

      {/* Overlays */}
      <DisconnectedOverlay onReconnect={reconnect} />
      {game.roundEndData && <RoundSummaryOverlay data={game.roundEndData} />}
      {game.gameOverData && (
        <GameOverOverlay
          data={game.gameOverData}
          localSeat={localSeat}
          onPlayAgain={playAgain}
          onLeave={leaveRoom}
        />
      )}
    </div>
  );
}
