import type { Phase } from "../../types";

const PHASE_NAMES: Record<number, string> = {
  0: "Bidding",
  1: "Choose Trump",
  2: "Playing",
};

interface PhaseIndicatorProps {
  phase: Phase;
}

export default function PhaseIndicator({ phase }: PhaseIndicatorProps) {
  return (
    <span className="badge bg-info" style={{ fontSize: "0.75rem" }}>
      {PHASE_NAMES[phase] ?? "Unknown"}
    </span>
  );
}
