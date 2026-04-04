import { useRef, useState, useEffect, useMemo } from "react";
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

const SUIT_ORDER: Record<number, number> = { 0: 0, 1: 1, 2: 2, 3: 3 };

function sortedHandIndices(cards: CardData[]): number[] {
  return cards
    .map((_, i) => i)
    .sort((a, b) => {
      const ca = cards[a];
      const cb = cards[b];
      // Jokers (suit === null) go last
      if (ca.suit === null && cb.suit !== null) return 1;
      if (ca.suit !== null && cb.suit === null) return -1;
      if (ca.suit === null && cb.suit === null) return 0;
      // Group by suit
      const suitDiff = SUIT_ORDER[ca.suit!] - SUIT_ORDER[cb.suit!];
      if (suitDiff !== 0) return suitDiff;
      // Within suit, rank descending (ace high)
      return cb.rank - ca.rank;
    });
}

function useCardDimensions() {
  const [dims, setDims] = useState({ width: 80, overlap: 28, height: 112 });
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const update = () => {
      const style = getComputedStyle(document.documentElement);
      const w = parseInt(style.getPropertyValue("--card-width")) || 80;
      const o = parseInt(style.getPropertyValue("--card-overlap")) || 28;
      const h = parseInt(style.getPropertyValue("--card-height")) || 112;
      setDims({ width: w, overlap: o, height: h });
    };
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  return { ref, ...dims };
}

export default function HandDisplay({ cards, faceUp, actionMask, isMyTurn, phase, onPlayCard }: HandDisplayProps) {
  const { width: cardWidth, overlap, height: cardHeight } = useCardDimensions();

  if (!faceUp) {
    if (cards.length === 0) return null;
    return (
      <div className="hand-display face-down">
        <div className="d-flex align-items-center gap-1">
          <Card faceUp={false} />
          {cards.length > 1 && (
            <span className="badge bg-secondary">{cards.length}</span>
          )}
        </div>
      </div>
    );
  }

  const displayOrder = useMemo(() => sortedHandIndices(cards), [cards]);
  const totalCards = cards.length;
  const totalWidth = totalCards > 0 ? cardWidth + (totalCards - 1) * overlap : 0;

  return (
    <div className="hand-display face-up" style={{ width: totalWidth, height: cardHeight + 8, position: "relative", margin: "0 auto" }}>
      {displayOrder.map((origIdx, displayIdx) => {
        const card = cards[origIdx];
        const isPlayable = isMyTurn && phase === 2 && actionMask?.[origIdx] === 1;
        const isDimmed = isMyTurn && phase === 2 && actionMask?.[origIdx] === 0;

        const midpoint = (totalCards - 1) / 2;
        const rotation = (displayIdx - midpoint) * 3;

        return (
          <Card
            key={`${card.suit}-${card.rank}-${origIdx}`}
            card={card}
            faceUp
            playable={isPlayable}
            dimmed={isDimmed}
            onClick={() => isPlayable && onPlayCard?.(origIdx)}
            style={{
              position: "absolute",
              left: displayIdx * overlap,
              transform: `rotate(${rotation}deg)`,
              transformOrigin: "bottom center",
              zIndex: displayIdx,
            }}
          />
        );
      })}
    </div>
  );
}
