import { useState } from "react";
import { Alert, Button, Form, Modal } from "react-bootstrap";

interface JoinGameModalProps {
  show: boolean;
  onBack: () => void;
  onJoin: (code: string) => void;
  loading: boolean;
  error: string;
}

export default function JoinGameModal({ show, onBack, onJoin, loading, error }: JoinGameModalProps) {
  const [code, setCode] = useState("");
  const isValid = code.length === 4;

  return (
    <Modal show={show} backdrop="static" centered>
      <Modal.Header>
        <Modal.Title>Join Game</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Form.Group className="mb-3">
          <Form.Label>Room Code</Form.Label>
          <Form.Control
            type="text"
            placeholder="ABCD"
            maxLength={4}
            value={code}
            onChange={(e) => setCode(e.target.value.toUpperCase())}
            onKeyDown={(e) => {
              if (e.key === "Enter" && isValid) onJoin(code);
            }}
            style={{ letterSpacing: "0.3em", textAlign: "center", fontSize: "1.5rem" }}
          />
        </Form.Group>
        {error && <Alert variant="danger">{error}</Alert>}
      </Modal.Body>
      <Modal.Footer>
        <Button variant="outline-secondary" onClick={onBack}>
          Back
        </Button>
        <Button variant="primary" onClick={() => onJoin(code)} disabled={!isValid || loading}>
          {loading ? "Joining..." : "Join"}
        </Button>
      </Modal.Footer>
    </Modal>
  );
}
