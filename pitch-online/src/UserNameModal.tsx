import { Button, Form, Modal } from "react-bootstrap";

interface UserNameModalProps {
  open: boolean;
  setOpen: React.Dispatch<React.SetStateAction<boolean>>;
  nameVal: string;
  setName: React.Dispatch<React.SetStateAction<string>>;
}

const UserNameModal = ({
  open,
  setOpen,
  nameVal,
  setName,
}: UserNameModalProps) => {
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
        <Form.Group className="mb-3" controlId="username">
          <Form.Label>username</Form.Label>
          <Form.Control
            type="text"
            placeholder="Enter name"
            value={nameVal}
            onChange={(val) => {
              setName(val.target.value);
            }}
          />
          <Form.Text className="text-muted">
            What do you want to be known as
          </Form.Text>
        </Form.Group>
        <div style={{ display: "flex", gap: "10px" }}>
          <Button variant="primary" style={{ width: "40%", margin: "auto" }}>
            Join a game
          </Button>
          <Button variant="primary" style={{ width: "40%", margin: "auto" }}>
            Create a new game
          </Button>
        </div>
      </Modal.Body>
    </Modal>
  );
};

export default UserNameModal;
