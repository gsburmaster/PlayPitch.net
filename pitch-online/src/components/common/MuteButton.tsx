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
      aria-label={muted ? "Unmute" : "Mute"}
      className="mute-btn"
    >
      {muted ? "\uD83D\uDD07" : "\uD83D\uDD0A"}
    </button>
  );
}
