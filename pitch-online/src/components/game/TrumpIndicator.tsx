import { useRef, useEffect, useState } from "react";
import { getSuitSymbol, getCardColor } from "../../types";

interface TrumpIndicatorProps {
  trumpSuit: number | null;
}

export default function TrumpIndicator({ trumpSuit }: TrumpIndicatorProps) {
  const prevSuit = useRef<number | null>(null);
  const [animating, setAnimating] = useState(false);

  useEffect(() => {
    if (trumpSuit !== null && prevSuit.current !== trumpSuit) {
      setAnimating(true);
      const timer = setTimeout(() => setAnimating(false), 500);
      prevSuit.current = trumpSuit;
      return () => clearTimeout(timer);
    }
    prevSuit.current = trumpSuit;
  }, [trumpSuit]);

  if (trumpSuit === null) {
    return (
      <div style={{ color: "rgba(255,255,255,0.4)", fontSize: "0.85rem" }}>
        Trump: --
      </div>
    );
  }

  const symbol = getSuitSymbol(trumpSuit);
  const color = getCardColor(trumpSuit);

  return (
    <div style={{ fontSize: "0.85rem", color: "white" }}>
      Trump:{" "}
      <span
        className={animating ? "anim-trump-reveal" : ""}
        style={{ color, fontSize: "1.2rem", display: "inline-block" }}
      >
        {symbol}
      </span>
    </div>
  );
}
