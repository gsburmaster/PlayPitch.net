import type { CardData } from "../../types";
import { getRankDisplay, getSuitSymbol, getCardColor, getCardPoints, SUIT_NAMES } from "../../types";
import "../../styles/card.css";

interface CardProps {
  card?: CardData;
  faceUp?: boolean;
  playable?: boolean;
  dimmed?: boolean;
  highlighted?: boolean;
  className?: string;
  onClick?: () => void;
  style?: React.CSSProperties;
}

export default function Card({ card, faceUp = true, playable, dimmed, highlighted, className, onClick, style }: CardProps) {
  if (!faceUp || !card) {
    return (
      <div className="pitch-card card-back" style={style}>
        <div className="card-back-pattern" />
      </div>
    );
  }

  const rank = getRankDisplay(card.rank);
  const suitSymbol = getSuitSymbol(card.suit);
  const color = getCardColor(card.suit);
  const isJoker = card.rank === 11;
  const points = getCardPoints(card.rank);

  const suitName = card.suit !== null ? SUIT_NAMES[card.suit] : "";
  const title = points
    ? `${rank} of ${suitName || "Joker"} — ${points}`
    : `${rank} of ${suitName || "Joker"}`;

  const classes = [
    "pitch-card",
    isJoker ? "joker" : color === "red" ? "red" : "black",
    playable ? "playable" : "",
    dimmed ? "dimmed" : "",
    highlighted ? "highlighted" : "",
    className ?? "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      className={classes}
      style={style}
      onClick={playable ? onClick : undefined}
      onKeyDown={playable ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick?.(); } } : undefined}
      role={playable ? "button" : undefined}
      tabIndex={playable ? 0 : undefined}
      title={title}
    >
      <div className="corner top-left">
        <span className="rank">{rank}</span>
        <span className="suit-symbol">{suitSymbol}</span>
      </div>
      <div className="center">
        <span className="suit-large">{suitSymbol}</span>
      </div>
      <div className="corner bottom-right rotated">
        <span className="rank">{rank}</span>
        <span className="suit-symbol">{suitSymbol}</span>
      </div>
    </div>
  );
}
