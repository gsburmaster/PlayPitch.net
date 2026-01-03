import { useState } from "react";
import { Button, Form, Modal } from "react-bootstrap";

interface JoinGameModalProps {
  open: boolean;
  setOpen: React.Dispatch<React.SetStateAction<boolean>>;
  handleJoinAttempt: (val: string) => void;
}

const StartingModal = ({
  open,
  setOpen,
  handleJoinAttempt,
}: JoinGameModalProps) => {
  const [gameKey, setGameKey] = useState<string>("");
  return (
    <Modal
      backdrop="static"
      show={open}
      onHide={() => {
        setOpen(false);
      }}
    >
      <Modal.Header>
        <h3>Pitch</h3>
      </Modal.Header>
      <Modal.Body>
        <Form.Label htmlFor="joinAGame">Join A Game</Form.Label>
        <Form.Control
          type="text"
          id="joinAGame"
          aria-describedby="joinAGameText"
          value={gameKey}
          onChange={(evt) => {
            setGameKey(evt.target.value);
          }}
        />
        <Form.Text id="joinAGameText" muted>
          Enter the game key to join a game
        </Form.Text>
        <Button
          onClick={() => {
            handleJoinAttempt(gameKey);
          }}
        >
          Join
        </Button>
        <Button></Button>
      </Modal.Body>
    </Modal>
  );
};

export default StartingModal;
