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

  return (
    <div className="action-panel text-center">
      <span className="text-white" style={{ fontSize: "0.85rem" }}>Click a card to play it</span>
    </div>
  );
}
