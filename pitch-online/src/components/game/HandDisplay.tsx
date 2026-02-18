import { useRef, useState, useEffect } from "react";
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

  const totalCards = cards.length;
  const totalWidth = totalCards > 0 ? cardWidth + (totalCards - 1) * overlap : 0;

  return (
    <div className="hand-display face-up" style={{ width: totalWidth, height: cardHeight + 8, position: "relative", margin: "0 auto" }}>
      {cards.map((card, idx) => {
        const isPlayable = isMyTurn && phase === 2 && actionMask?.[idx] === 1;
        const isDimmed = isMyTurn && phase === 2 && actionMask?.[idx] === 0;

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
