import { Button } from "react-bootstrap";

interface SuitPickerProps {
  onAction: (action: number) => void;
}

const SUITS = [
  { action: 19, symbol: "\u2665", label: "Hearts", className: "suit-btn suit-btn--red" },
  { action: 20, symbol: "\u2666", label: "Diamonds", className: "suit-btn suit-btn--red" },
  { action: 21, symbol: "\u2663", label: "Clubs", className: "suit-btn suit-btn--dark" },
  { action: 22, symbol: "\u2660", label: "Spades", className: "suit-btn suit-btn--dark" },
];

export default function SuitPicker({ onAction }: SuitPickerProps) {
  return (
    <div className="action-panel text-center">
      <div className="mb-2 text-white">Choose your trump suit</div>
      <div className="d-flex gap-2 justify-content-center">
        {SUITS.map(({ action, symbol, label, className }) => (
          <Button
            key={action}
            variant="light"
            onClick={() => onAction(action)}
            className={className}
            title={label}
          >
            {symbol}
          </Button>
        ))}
      </div>
    </div>
  );
}
