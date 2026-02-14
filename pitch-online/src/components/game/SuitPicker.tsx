import { Button } from "react-bootstrap";

interface SuitPickerProps {
  onAction: (action: number) => void;
}

const SUITS = [
  { action: 19, symbol: "\u2665", label: "Hearts", color: "#dc3545" },
  { action: 20, symbol: "\u2666", label: "Diamonds", color: "#dc3545" },
  { action: 21, symbol: "\u2663", label: "Clubs", color: "#333" },
  { action: 22, symbol: "\u2660", label: "Spades", color: "#333" },
];

export default function SuitPicker({ onAction }: SuitPickerProps) {
  return (
    <div className="action-panel text-center">
      <div className="mb-2 text-white">Choose your trump suit</div>
      <div className="d-flex gap-2 justify-content-center">
        {SUITS.map(({ action, symbol, label, color }) => (
          <Button
            key={action}
            variant="light"
            onClick={() => onAction(action)}
            style={{ fontSize: "1.5rem", color, minWidth: 60 }}
            title={label}
          >
            {symbol}
          </Button>
        ))}
      </div>
    </div>
  );
}
