import { useState } from "react";
import { Button, Form, Modal, Spinner } from "react-bootstrap";

interface CreateGameModalProps {
  show: boolean;
  onBack: () => void;
  onCreate: (aiSeats: number[]) => void;
  loading: boolean;
}

const TEAM_LABELS = ["Team A", "Team B", "Team A", "Team B"];
const POSITIONS = ["Bottom", "Left", "Top", "Right"];

export default function CreateGameModal({ show, onBack, onCreate, loading }: CreateGameModalProps) {
  // seats 1, 2, 3 — true means AI, false means open for human
  const [seatIsAI, setSeatIsAI] = useState<Record<number, boolean>>({ 1: true, 2: true, 3: true });

  const toggleSeat = (seat: number) => {
    setSeatIsAI((prev) => ({ ...prev, [seat]: !prev[seat] }));
  };

  const aiSeats = [1, 2, 3].filter((s) => seatIsAI[s]);
  const openCount = 3 - aiSeats.length;

  const helperText =
    openCount === 0
      ? "You + 3 AI"
      : openCount === 3
        ? "You + 3 open seats"
        : `You + ${aiSeats.length} AI + ${openCount} open seat${openCount > 1 ? "s" : ""}`;

  return (
    <Modal show={show} backdrop="static" centered>
      <Modal.Header>
        <Modal.Title>Create Game</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <label className="form-label d-block mb-2">Arrange seats</label>
        <p className="text-muted small mb-3">Click a seat to toggle between AI and open (for another player). Teammates sit across from each other.</p>
        <div className="d-flex flex-column align-items-center gap-2 mb-3">
          {/* Top seat: seat 2 (Team A — your teammate) */}
          <SeatButton seat={2} isAI={seatIsAI[2]} team={TEAM_LABELS[2]} position={POSITIONS[2]} onClick={toggleSeat} />
          {/* Middle row: seat 1 (left) and seat 3 (right) */}
          <div className="d-flex justify-content-center gap-4 w-100">
            <SeatButton seat={1} isAI={seatIsAI[1]} team={TEAM_LABELS[1]} position={POSITIONS[1]} onClick={toggleSeat} />
            <SeatButton seat={3} isAI={seatIsAI[3]} team={TEAM_LABELS[3]} position={POSITIONS[3]} onClick={toggleSeat} />
          </div>
          {/* Bottom seat: seat 0 (you) — not toggleable */}
          <div
            className="rounded p-2 text-center"
            style={{
              border: "2px solid #28a745",
              backgroundColor: "rgba(40,167,69,0.15)",
              minWidth: 130,
              color: "white",
            }}
          >
            <div style={{ fontWeight: "bold" }}>You</div>
            <small className="text-muted">{POSITIONS[0]} - {TEAM_LABELS[0]}</small>
          </div>
        </div>
        <Form.Text className="d-block text-center text-muted">{helperText}</Form.Text>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="outline-secondary" onClick={onBack}>
          Back
        </Button>
        <Button variant="success" onClick={() => onCreate(aiSeats)} disabled={loading}>
          {loading ? <><Spinner animation="border" size="sm" className="me-2" />Creating...</> : "Create"}
        </Button>
      </Modal.Footer>
    </Modal>
  );
}

function SeatButton({ seat, isAI, team, position, onClick }: { seat: number; isAI: boolean; team: string; position: string; onClick: (seat: number) => void }) {
  return (
    <div
      className="rounded p-2 text-center"
      style={{
        border: `2px solid ${isAI ? "#ffc107" : "rgba(255,255,255,0.3)"}`,
        backgroundColor: isAI ? "rgba(255,193,7,0.12)" : "rgba(0,0,0,0.3)",
        minWidth: 130,
        cursor: "pointer",
        color: "white",
        userSelect: "none",
      }}
      onClick={() => onClick(seat)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") onClick(seat); }}
    >
      <div style={{ fontWeight: "bold" }}>{isAI ? "AI" : "Open"}</div>
      <small className="text-muted">{position} - {team}</small>
    </div>
  );
}
