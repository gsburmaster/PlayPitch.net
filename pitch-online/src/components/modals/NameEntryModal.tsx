import { useState } from "react";
import { Button, Form, Modal } from "react-bootstrap";
import HowToPlayModal from "./HowToPlayModal";

interface NameEntryModalProps {
  show: boolean;
  onCreateGame: (name: string) => void;
  onJoinGame: (name: string) => void;
}

export default function NameEntryModal({ show, onCreateGame, onJoinGame }: NameEntryModalProps) {
  const [name, setName] = useState("");
  const [showRules, setShowRules] = useState(false);
  const isValid = name.trim().length >= 1 && name.length <= 16;

  return (
    <>
      <Modal show={show && !showRules} backdrop="static" centered>
        <Modal.Header>
          <Modal.Title>Pitch</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form.Group className="mb-3">
            <Form.Label>Display Name</Form.Label>
            <Form.Control
              type="text"
              placeholder="Enter your name"
              maxLength={16}
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && isValid) onCreateGame(name.trim());
              }}
            />
          </Form.Group>
          <div className="d-flex gap-2 justify-content-center">
            <Button variant="success" disabled={!isValid} onClick={() => onCreateGame(name.trim())}>
              Create Game
            </Button>
            <Button variant="primary" disabled={!isValid} onClick={() => onJoinGame(name.trim())}>
              Join Game
            </Button>
          </div>
        </Modal.Body>
        <Modal.Footer className="justify-content-center border-0 pt-0">
          <Button variant="link" className="text-muted" onClick={() => setShowRules(true)}>
            How to Play
          </Button>
        </Modal.Footer>
      </Modal>
      <HowToPlayModal show={showRules} onClose={() => setShowRules(false)} />
    </>
  );
}
