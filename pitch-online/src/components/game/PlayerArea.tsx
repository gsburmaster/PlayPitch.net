import type { CardData, SeatIndex } from "../../types";
import type { BidEntry } from "../../contexts/GameContext";
import PlayerLabel from "./PlayerLabel";
import HandDisplay from "./HandDisplay";
import BidBubble from "./BidBubble";

interface PlayerAreaProps {
  seatIndex: SeatIndex;
  displayName: string;
  isLocal: boolean;
  isDealer: boolean;
  isConnected: boolean;
  isCurrentPlayer: boolean;
  cards: CardData[];
  cardCount: number;
  bidEntry?: BidEntry;
  actionMask?: number[] | null;
  isMyTurn?: boolean;
  phase?: number;
  onPlayCard?: (index: number) => void;
  position: "bottom" | "left" | "top" | "right";
}

export default function PlayerArea({
  displayName,
  isLocal,
  isDealer,
  isConnected,
  isCurrentPlayer,
  cards,
  cardCount,
  bidEntry,
  actionMask,
  isMyTurn,
  phase,
  onPlayCard,
  position,
}: PlayerAreaProps) {
  const faceUp = isLocal;
  const displayCards = faceUp ? cards : Array(cardCount).fill({ suit: null, rank: 0 });

  return (
    <div className={`player-area player-area-${position}`}>
      <PlayerLabel
        name={displayName}
        isDealer={isDealer}
        isConnected={isConnected}
        isLocal={isLocal}
        isCurrentPlayer={isCurrentPlayer}
      />
      <HandDisplay
        cards={displayCards}
        faceUp={faceUp}
        actionMask={isLocal ? actionMask : null}
        isMyTurn={isLocal && isMyTurn}
        phase={phase}
        onPlayCard={isLocal ? onPlayCard : undefined}
      />
      {bidEntry && <BidBubble action={bidEntry.action} displayName={bidEntry.displayName} />}
    </div>
  );
}
