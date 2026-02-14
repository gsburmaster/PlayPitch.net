import { useState } from "react";
import { Button, Card, Spinner } from "react-bootstrap";
import { useAppState } from "../contexts/AppContext";
import { useGameState } from "../contexts/GameContext";
import { useWebSocket } from "../hooks/useWebSocket";
import RoomCodeDisplay from "../components/lobby/RoomCodeDisplay";
import SeatMap from "../components/lobby/SeatMap";
import DisconnectedOverlay from "../components/common/DisconnectedOverlay";
import "../styles/table.css";

export default function LobbyScreen() {
  const { roomCode, seatIndex, isCreator } = useAppState();
  const { seats } = useGameState();
  const { startGame, leaveRoom, reconnect } = useWebSocket();
  const [starting, setStarting] = useState(false);

  const allFilled = seats.length === 4;

  const handleStart = () => {
    setStarting(true);
    startGame();
  };

  return (
    <div className="table-bg d-flex align-items-center justify-content-center" style={{ minHeight: "100vh" }}>
      <DisconnectedOverlay onReconnect={reconnect} />
      <Card style={{ maxWidth: 500, width: "100%", backgroundColor: "rgba(0,50,0,0.85)", color: "white", border: "1px solid rgba(255,255,255,0.2)" }}>
        <Card.Body>
          <RoomCodeDisplay code={roomCode} />
          <SeatMap seats={seats} localSeat={seatIndex} />
          <div className="d-flex gap-2 justify-content-center mt-3">
            {isCreator && (
              <Button variant="success" disabled={!allFilled || starting} onClick={handleStart}>
                {starting ? <><Spinner animation="border" size="sm" className="me-2" />Starting...</> : "Start Game"}
              </Button>
            )}
            <Button variant="outline-danger" onClick={leaveRoom}>
              Leave Room
            </Button>
          </div>
          {!allFilled && (
            <p className="text-center text-muted mt-2 mb-0">
              Waiting for {4 - seats.length} more player{4 - seats.length !== 1 ? "s" : ""}...
            </p>
          )}
        </Card.Body>
      </Card>
    </div>
  );
}
