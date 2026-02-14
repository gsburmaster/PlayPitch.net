import { useState } from "react";
import type { TrickResultData } from "../../state/gameReducer";
import type { SeatIndex } from "../../types";
import { getRankDisplay, getSuitSymbol } from "../../types";

interface TrickHistoryDrawerProps {
  trickHistory: TrickResultData[];
  localSeat: SeatIndex;
  seatNames: Record<number, string>;
}

export default function TrickHistoryDrawer({ trickHistory, localSeat, seatNames }: TrickHistoryDrawerProps) {
  const [open, setOpen] = useState(false);

  const trickCount = trickHistory.length;

  return (
    <>
      {/* Toggle button */}
      <button
        onClick={() => setOpen(!open)}
        title="Trick History"
        className={`trick-btn${open ? " trick-btn--open" : ""}`}
      >
        Tricks {trickCount > 0 ? `(${trickCount})` : ""}
      </button>

      {/* Drawer panel */}
      {open && (
        <div className="drawer">
          {/* Header */}
          <div className="drawer-header">
            <span className="drawer-header-title">
              Trick History
            </span>
            <button
              onClick={() => setOpen(false)}
              className="drawer-close"
            >
              ×
            </button>
          </div>

          {/* Trick list */}
          <div className="drawer-body">
            {trickCount === 0 ? (
              <div className="drawer-empty">
                No tricks played yet
              </div>
            ) : (
              trickHistory.map((trick, idx) => (
                <TrickEntry
                  key={idx}
                  trickNumber={idx + 1}
                  trick={trick}
                  localSeat={localSeat}
                  seatNames={seatNames}
                />
              ))
            )}
          </div>
        </div>
      )}

      {/* Backdrop */}
      {open && (
        <div
          onClick={() => setOpen(false)}
          className="drawer-backdrop"
        />
      )}
    </>
  );
}

function TrickEntry({
  trickNumber,
  trick,
  localSeat,
  seatNames,
}: {
  trickNumber: number;
  trick: TrickResultData;
  localSeat: SeatIndex;
  seatNames: Record<number, string>;
}) {
  return (
    <div className="trick-entry">
      {/* Trick header */}
      <div className="trick-entry-header">
        <span className="trick-entry-number">
          Trick {trickNumber}
        </span>
        {trick.pointsWon > 0 && (
          <span className="trick-entry-points">
            +{trick.pointsWon} pt{trick.pointsWon !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {/* Cards played */}
      <div className="trick-card-row">
        {trick.trick.map((entry, i) => {
          const isWinner = entry.seatIndex === trick.winner;
          const isLocal = entry.seatIndex === localSeat;
          const name = isLocal ? "You" : (seatNames[entry.seatIndex] ?? `P${entry.seatIndex}`);
          const rank = getRankDisplay(entry.card.rank);
          const suit = getSuitSymbol(entry.card.suit);
          const color = entry.card.suit !== null
            ? (entry.card.suit === 0 || entry.card.suit === 1 ? "#dc3545" : "#ccc")
            : "#b366cc";

          return (
            <div
              key={i}
              className={`trick-card-line${isWinner ? " trick-card-line--winner" : ""}`}
            >
              {isWinner && <span style={{ fontSize: "0.65rem" }}>★</span>}
              <span className={`trick-card-name${isLocal ? " trick-card-name--local" : ""}`}>
                {name}
              </span>
              <span style={{ color }}>{rank}{suit}</span>
            </div>
          );
        })}
      </div>

      {/* Winner line */}
      <div className="trick-winner-line">
        Won by {trick.winner === localSeat ? "You" : trick.winnerName}
      </div>
    </div>
  );
}
