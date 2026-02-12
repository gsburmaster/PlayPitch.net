import type { CardData } from "../../types";
import Card from "./Card";

interface HandDisplayProps {
  cards: CardData[];
  faceUp: boolean;
  actionMask?: number[] | null;
  isMyTurn?: boolean;
  phase?: number;
  onPlayCard?: (index: number) => void;
}

export default function HandDisplay({ cards, faceUp, actionMask, isMyTurn, phase, onPlayCard }: HandDisplayProps) {
  if (!faceUp) {
    // Show card count badge
    return (
      <div className="hand-display face-down">
        <div className="d-flex align-items-center gap-1">
          <Card faceUp={false} />
          {cards.length > 1 && (
            <span className="badge bg-dark">{cards.length}</span>
          )}
        </div>
      </div>
    );
  }

  const totalCards = cards.length;
  const cardWidth = 80;
  const overlap = 28;
  const totalWidth = totalCards > 0 ? cardWidth + (totalCards - 1) * overlap : 0;

  return (
    <div className="hand-display face-up" style={{ width: totalWidth, height: 120, position: "relative", margin: "0 auto" }}>
      {cards.map((card, idx) => {
        const isPlayable = isMyTurn && phase === 2 && actionMask?.[idx] === 1;
        const isDimmed = isMyTurn && phase === 2 && actionMask?.[idx] === 0;

        // Fan arc: slight rotation
        const midpoint = (totalCards - 1) / 2;
        const rotation = (idx - midpoint) * 3;

        return (
          <Card
            key={`${card.suit}-${card.rank}-${idx}`}
            card={card}
            faceUp
            playable={isPlayable}
            dimmed={isDimmed}
            onClick={() => isPlayable && onPlayCard?.(idx)}
            style={{
              position: "absolute",
              left: idx * overlap,
              transform: `rotate(${rotation}deg)`,
              transformOrigin: "bottom center",
              zIndex: idx,
            }}
          />
        );
      })}
    </div>
  );
}
