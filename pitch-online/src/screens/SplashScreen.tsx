import { useState } from "react";
import NameEntryModal from "../components/modals/NameEntryModal";
import CreateGameModal from "../components/modals/CreateGameModal";
import JoinGameModal from "../components/modals/JoinGameModal";
import { useAppDispatch } from "../contexts/AppContext";
import type { SeatIndex } from "../types";
import "../styles/splash.css";

const API_BASE = import.meta.env.VITE_API_URL || "";

type ModalView = "name" | "create" | "join";

export default function SplashScreen() {
  const dispatch = useAppDispatch();
  const [modalView, setModalView] = useState<ModalView>("name");
  const [displayName, setDisplayName] = useState("");
  const [loading, setLoading] = useState(false);
  const [joinError, setJoinError] = useState("");

  const handleCreateGame = async (aiSeats: number[]) => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/rooms`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ displayName, aiSeats }),
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
      setJoinError("Failed to create room");
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
        body: JSON.stringify({ displayName }),
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
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="splash-bg">
      <NameEntryModal
        show={modalView === "name"}
        onCreateGame={(name) => {
          setDisplayName(name);
          setModalView("create");
        }}
        onJoinGame={(name) => {
          setDisplayName(name);
          setModalView("join");
        }}
      />
      <CreateGameModal
        show={modalView === "create"}
        onBack={() => setModalView("name")}
        onCreate={handleCreateGame}
        loading={loading}
      />
      <JoinGameModal
        show={modalView === "join"}
        onBack={() => {
          setJoinError("");
          setModalView("name");
        }}
        onJoin={handleJoinGame}
        loading={loading}
        error={joinError}
      />
    </div>
  );
}
