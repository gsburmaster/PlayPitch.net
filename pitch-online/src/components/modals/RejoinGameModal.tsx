import { Button, Modal } from "react-bootstrap";

interface RejoinGameModalProps {
  show: boolean;
  roomCode: string;
  onRejoin: () => void;
  onDismiss: () => void;
}

export default function RejoinGameModal({ show, roomCode, onRejoin, onDismiss }: RejoinGameModalProps) {
  return (
    <Modal show={show} backdrop="static" centered>
      <Modal.Header>
        <Modal.Title>Rejoin Game?</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <p>
          You have an active game in room <strong>{roomCode}</strong>. Would you like to rejoin?
        </p>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="outline-secondary" onClick={onDismiss}>
          No Thanks
        </Button>
        <Button variant="success" onClick={onRejoin}>
          Rejoin
        </Button>
      </Modal.Footer>
    </Modal>
  );
}
