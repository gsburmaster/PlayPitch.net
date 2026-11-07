import { Button, Card } from "react-bootstrap";
import { useAppState } from "../contexts/AppContext";
import { useGameState } from "../contexts/GameContext";
import { useWebSocket } from "../contexts/WebSocketContext";
import RoomCodeDisplay from "../components/lobby/RoomCodeDisplay";
import SeatMap from "../components/lobby/SeatMap";
import DisconnectedOverlay from "../components/common/DisconnectedOverlay";
import "../styles/table.css";

export default function LobbyScreen() {
  const { roomCode, seatIndex, isCreator } = useAppState();
  const { seats } = useGameState();
  const { startGame, leaveRoom, reconnect } = useWebSocket();

  const allFilled = seats.length === 4;

  return (
    <div className="table-bg d-flex align-items-center justify-content-center" style={{ minHeight: "100vh" }}>
      <DisconnectedOverlay onReconnect={reconnect} />
      <Card className="lobby-card">
        <Card.Body>
          <RoomCodeDisplay code={roomCode} />
          <SeatMap seats={seats} localSeat={seatIndex} />
          <div className="d-flex gap-2 justify-content-center mt-3">
            {/* Starting twice is harmless: the server ignores a start once the game is on. */}
            {isCreator && (
              <Button variant="success" disabled={!allFilled} onClick={startGame}>
                Start Game
              </Button>
            )}
            <Button variant="outline-danger" onClick={leaveRoom}>
              Leave Room
            </Button>
          </div>
          {!allFilled && (
            <p className="text-center lobby-muted mt-2 mb-0">
              Waiting for {4 - seats.length} more player{4 - seats.length !== 1 ? "s" : ""}...
            </p>
          )}
        </Card.Body>
      </Card>
    </div>
  );
}
