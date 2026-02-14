import { Button } from "react-bootstrap";

interface PlayPromptProps {
  actionMask: number[];
  onAction: (action: number) => void;
}

export default function PlayPrompt({ actionMask, onAction }: PlayPromptProps) {
  // If action_mask[23] == 1, no valid plays
  if (actionMask[23] === 1) {
    return (
      <div className="action-panel text-center">
        <Button variant="warning" onClick={() => onAction(23)}>
          No Valid Plays
        </Button>
      </div>
    );
  }

  // Check if some cards are dimmed (not all playable)
  const playableCount = actionMask.slice(0, 10).filter((v) => v === 1).length;
  const totalCards = actionMask.slice(0, 10).filter((v) => v === 0 || v === 1).length;
  const hasFiltered = playableCount < totalCards && playableCount > 0;

  return (
    <div className="action-panel text-center">
      <span className="text-white" style={{ fontSize: "0.85rem" }}>
        Click a card to play it
      </span>
      {hasFiltered && (
        <div className="text-warning mt-1" style={{ fontSize: "0.75rem", opacity: 0.8 }}>
          Only trump cards can be played
        </div>
      )}
    </div>
  );
}
