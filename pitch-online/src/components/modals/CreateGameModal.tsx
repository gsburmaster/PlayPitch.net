import { useState } from "react";
import { Button, Form, Modal, ToggleButton, ToggleButtonGroup } from "react-bootstrap";

interface CreateGameModalProps {
  show: boolean;
  onBack: () => void;
  onCreate: (aiCount: number) => void;
  loading: boolean;
}

export default function CreateGameModal({ show, onBack, onCreate, loading }: CreateGameModalProps) {
  const [aiCount, setAiCount] = useState(3);

  const openSeats = 3 - aiCount;
  const helperText =
    aiCount === 3
      ? "You + 3 AI"
      : aiCount === 0
        ? "You + 3 open seats"
        : `You + ${openSeats} open seat${openSeats > 1 ? "s" : ""} + ${aiCount} AI`;

  return (
    <Modal show={show} backdrop="static" centered>
      <Modal.Header>
        <Modal.Title>Create Game</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <div className="mb-3">
          <label className="form-label d-block">How many AI players?</label>
          <ToggleButtonGroup
            type="radio"
            name="aiCount"
            value={aiCount}
            onChange={(val: number) => setAiCount(val)}
          >
            {[0, 1, 2, 3].map((n) => (
              <ToggleButton key={n} id={`ai-${n}`} value={n} variant="outline-success">
                {n}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
          <Form.Text className="d-block mt-1 text-muted">{helperText}</Form.Text>
        </div>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="outline-secondary" onClick={onBack}>
          Back
        </Button>
        <Button variant="success" onClick={() => onCreate(aiCount)} disabled={loading}>
          {loading ? "Creating..." : "Create"}
        </Button>
      </Modal.Footer>
    </Modal>
  );
}

