import { useState } from "react";
import { useSound } from "../../hooks/useSound";

export default function MuteButton() {
  const { setMuted, isMuted } = useSound();
  const [muted, setMutedState] = useState(isMuted());

  const toggle = () => {
    const next = !muted;
    setMuted(next);
    setMutedState(next);
  };

  return (
    <button
      onClick={toggle}
      title={muted ? "Unmute" : "Mute"}
      style={{
        position: "fixed",
        bottom: 10,
        right: 10,
        zIndex: 9999,
        width: 36,
        height: 36,
        borderRadius: "50%",
        border: "1px solid rgba(255,255,255,0.3)",
        backgroundColor: "rgba(0,0,0,0.5)",
        color: "white",
        fontSize: "1.1rem",
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 0,
      }}
    >
      {muted ? "\uD83D\uDD07" : "\uD83D\uDD0A"}
    </button>
  );
}
