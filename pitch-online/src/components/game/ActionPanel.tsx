import type { SeatInfo } from "../../types";
import BiddingControls from "./BiddingControls";
import SuitPicker from "./SuitPicker";
import PlayPrompt from "./PlayPrompt";

interface ActionPanelProps {
  phase: number;
  actionMask: number[] | null;
  isMyTurn: boolean;
  currentBid: number;
  currentHighBidder: number;
  seats: SeatInfo[];
  trumpChooserName?: string;
  onAction: (action: number) => void;
}

export default function ActionPanel({
  phase,
  actionMask,
  isMyTurn,
  currentBid,
  currentHighBidder,
  seats,
  trumpChooserName,
  onAction,
}: ActionPanelProps) {
  if (!isMyTurn || !actionMask) {
    // Show waiting text
    if (phase === 1 && trumpChooserName) {
      return (
        <div className="action-panel text-center text-white" style={{ fontSize: "0.85rem" }}>
          Waiting for {trumpChooserName} to choose trump...
        </div>
      );
    }
    return null;
  }

  const bidderName = seats.find((s) => s.seatIndex === currentHighBidder)?.displayName ?? "";

  switch (phase) {
    case 0:
      return (
        <BiddingControls
          actionMask={actionMask}
          currentBid={currentBid}
          currentHighBidder={currentHighBidder}
          bidderName={bidderName}
          onAction={onAction}
        />
      );
    case 1:
      return <SuitPicker onAction={onAction} />;
    case 2:
      return <PlayPrompt actionMask={actionMask} onAction={onAction} />;
    default:
      return null;
  }
}
