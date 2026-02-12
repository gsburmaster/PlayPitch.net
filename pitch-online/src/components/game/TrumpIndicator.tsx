import { getSuitSymbol, getCardColor } from "../../types";

interface TrumpIndicatorProps {
  trumpSuit: number | null;
}

export default function TrumpIndicator({ trumpSuit }: TrumpIndicatorProps) {
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
      Trump: <span style={{ color, fontSize: "1.2rem" }}>{symbol}</span>
    </div>
  );
}
