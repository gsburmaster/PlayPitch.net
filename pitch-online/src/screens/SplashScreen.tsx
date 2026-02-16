import { useState } from "react";
import { Button } from "react-bootstrap";
import CreateGameModal from "../components/modals/CreateGameModal";
import JoinGameModal from "../components/modals/JoinGameModal";
import HowToPlayModal from "../components/modals/HowToPlayModal";
import PrivacyPolicyModal from "../components/modals/PrivacyPolicyModal";
import { useAppDispatch } from "../contexts/AppContext";
import { useToast } from "../components/common/Toast";
import type { SeatIndex } from "../types";
import "../styles/splash.css";

const API_BASE = import.meta.env.VITE_API_URL || "";

type ModalView = "none" | "create" | "join" | "rules" | "privacy";

export default function SplashScreen() {
  const dispatch = useAppDispatch();
  const { addToast } = useToast();
  const [modalView, setModalView] = useState<ModalView>("none");
  const [displayName, setDisplayName] = useState("");
  const [loading, setLoading] = useState(false);
  const [joinError, setJoinError] = useState("");

  const isValid = displayName.trim().length >= 1 && displayName.length <= 16;

  const handleCreateGame = async (aiSeats: number[]) => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/rooms`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ displayName: displayName.trim(), aiSeats }),
      });
      if (!res.ok) throw new Error("Failed to create room");
      const data = await res.json();
      dispatch({
        type: "ROOM_JOINED",
        playerId: data.playerId,
        roomCode: data.roomCode,
        seatIndex: data.seatIndex as SeatIndex,
        isCreator: true,
      });
    } catch {
      addToast("Failed to create room", "error");
    } finally {
      setLoading(false);
    }
  };

  const handleJoinGame = async (code: string) => {
    setLoading(true);
    setJoinError("");
    try {
      const res = await fetch(`${API_BASE}/api/rooms/${code}/join`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ displayName: displayName.trim() }),
      });
      if (!res.ok) {
        const err = await res.json();
        setJoinError(err.error || "Failed to join");
        return;
      }
      const data = await res.json();
      dispatch({
        type: "ROOM_JOINED",
        playerId: data.playerId,
        roomCode: data.roomCode,
        seatIndex: data.seatIndex as SeatIndex,
        isCreator: false,
      });
    } catch {
      setJoinError("Failed to connect to server");
      addToast("Failed to connect to server", "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="splash-bg">
      <h1 className="splash-title">Pitch</h1>
      <p className="splash-subtitle">The classic trick-taking card game</p>

      <div className="splash-panel">
        <label className="splash-input-label">Display Name</label>
        <input
          className="splash-input"
          type="text"
          placeholder="Enter your name"
          maxLength={16}
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && isValid) setModalView("create");
          }}
        />
        <div className="splash-actions">
          <Button variant="success" disabled={!isValid} onClick={() => setModalView("create")}>
            Create Game
          </Button>
          <Button variant="primary" disabled={!isValid} onClick={() => setModalView("join")}>
            Join Game
          </Button>
        </div>
      </div>

      <div className="splash-links">
        <button className="splash-link" onClick={() => setModalView("rules")}>
          How to Play
        </button>
        <button className="splash-link" onClick={() => setModalView("privacy")}>
          Privacy Policy
        </button>
      </div>

      <CreateGameModal
        show={modalView === "create"}
        onBack={() => setModalView("none")}
        onCreate={handleCreateGame}
        loading={loading}
      />
      <JoinGameModal
        show={modalView === "join"}
        onBack={() => {
          setJoinError("");
          setModalView("none");
        }}
        onJoin={handleJoinGame}
        loading={loading}
        error={joinError}
      />
      <HowToPlayModal
        show={modalView === "rules"}
        onClose={() => setModalView("none")}
      />
      <PrivacyPolicyModal
        show={modalView === "privacy"}
        onClose={() => setModalView("none")}
      />
    </div>
  );
}
