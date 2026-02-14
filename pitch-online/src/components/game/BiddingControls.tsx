import { Button } from "react-bootstrap";
import { bidDisplay } from "../../types";

interface BiddingControlsProps {
  actionMask: number[];
  currentBid: number;
  currentHighBidder: number;
  bidderName: string;
  onAction: (action: number) => void;
}

export default function BiddingControls({ actionMask, currentBid, bidderName, onAction }: BiddingControlsProps) {
  const bidText = currentBid > 0 ? `Current bid: ${bidDisplay(currentBid)} by ${bidderName}` : "No bids yet";

  // Action mapping: 10=pass, 11=5, 12=6, ..., 16=10, 17=Moon, 18=DblMoon
  const bidButtons = [
    { action: 11, label: "5" },
    { action: 12, label: "6" },
    { action: 13, label: "7" },
    { action: 14, label: "8" },
    { action: 15, label: "9" },
    { action: 16, label: "10" },
    { action: 17, label: "Moon" },
    { action: 18, label: "Dbl Moon" },
  ];

  return (
    <div className="action-panel text-center">
      <div className="mb-2 text-white bid-text">{bidText}</div>
      <div className="d-flex flex-wrap gap-1 justify-content-center">
        <Button
          variant="outline-secondary"
          size="sm"
          disabled={actionMask[10] !== 1}
          onClick={() => onAction(10)}
        >
          Pass
        </Button>
        {bidButtons.map(({ action, label }) =>
          actionMask[action] === 1 ? (
            <Button key={action} variant="success" size="sm" onClick={() => onAction(action)}>
              {label}
            </Button>
          ) : null
        )}
      </div>
    </div>
  );
}
