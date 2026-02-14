import { createContext, useContext, useCallback, useRef, useMemo } from "react";

export type SoundName = "cardPlay" | "trickWon" | "bid" | "pass" | "roundEnd" | "gameWin" | "gameLose" | "trumpChosen" | "yourTurn";

interface SoundContextValue {
  play: (name: SoundName) => void;
  setMuted: (muted: boolean) => void;
  isMuted: () => boolean;
}

let audioCtx: AudioContext | null = null;

function getCtx(): AudioContext | null {
  if (!audioCtx) {
    try {
      audioCtx = new AudioContext();
    } catch {
      return null;
    }
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
  return audioCtx;
}

function playTone(freq: number, duration: number, type: OscillatorType = "sine", volume = 0.15) {
  const ctx = getCtx();
  if (!ctx) return;

  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(volume, ctx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start(ctx.currentTime);
  osc.stop(ctx.currentTime + duration);
}

function playChord(freqs: number[], duration: number, type: OscillatorType = "sine", volume = 0.08) {
  for (const f of freqs) playTone(f, duration, type, volume);
}

const SOUNDS: Record<SoundName, () => void> = {
  cardPlay: () => {
    // Quick snap sound
    playTone(800, 0.08, "square", 0.06);
    setTimeout(() => playTone(400, 0.05, "square", 0.04), 30);
  },
  trickWon: () => {
    // Rising two-note chime
    playTone(523, 0.15, "sine", 0.12);
    setTimeout(() => playTone(659, 0.2, "sine", 0.12), 120);
  },
  bid: () => {
    // Single confident tone
    playTone(440, 0.12, "triangle", 0.1);
  },
  pass: () => {
    // Soft descending tone
    playTone(330, 0.1, "sine", 0.06);
  },
  roundEnd: () => {
    // Three ascending notes
    playTone(523, 0.15, "sine", 0.1);
    setTimeout(() => playTone(659, 0.15, "sine", 0.1), 150);
    setTimeout(() => playTone(784, 0.25, "sine", 0.1), 300);
  },
  gameWin: () => {
    // Major chord fanfare
    setTimeout(() => playChord([523, 659, 784], 0.4, "sine", 0.1), 0);
    setTimeout(() => playChord([587, 740, 880], 0.5, "sine", 0.1), 350);
    setTimeout(() => playChord([659, 784, 1047], 0.7, "sine", 0.12), 700);
  },
  gameLose: () => {
    // Minor descending
    playTone(440, 0.3, "sine", 0.1);
    setTimeout(() => playTone(370, 0.3, "sine", 0.1), 250);
    setTimeout(() => playTone(330, 0.5, "sine", 0.1), 500);
  },
  trumpChosen: () => {
    // Quick flourish
    playTone(660, 0.1, "triangle", 0.1);
    setTimeout(() => playTone(880, 0.15, "triangle", 0.1), 80);
  },
  yourTurn: () => {
    // Gentle notification
    playTone(587, 0.12, "sine", 0.08);
    setTimeout(() => playTone(784, 0.15, "sine", 0.08), 100);
  },
};

let muted = false;

export const SoundContext = createContext<SoundContextValue>({
  play: () => {},
  setMuted: () => {},
  isMuted: () => false,
});

export function useSoundSystem(): SoundContextValue {
  const mutedRef = useRef(muted);

  const play = useCallback((name: SoundName) => {
    if (mutedRef.current) return;
    SOUNDS[name]?.();
  }, []);

  const setMuted = useCallback((m: boolean) => {
    muted = m;
    mutedRef.current = m;
  }, []);

  const isMuted = useCallback(() => mutedRef.current, []);

  return useMemo(() => ({ play, setMuted, isMuted }), [play, setMuted, isMuted]);
}

export function useSound() {
  return useContext(SoundContext);
}
