import { useState } from "react";
import type { TrickResultData } from "../../state/gameReducer";
import type { SeatIndex } from "../../types";
import { getRankDisplay, getSuitSymbol, getCardColor } from "../../types";

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
        style={{
          backgroundColor: open ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.4)",
          color: "white",
          border: "1px solid rgba(255,255,255,0.25)",
          borderRadius: 6,
          padding: "4px 10px",
          fontSize: "0.75rem",
          cursor: "pointer",
          whiteSpace: "nowrap",
        }}
      >
        Tricks {trickCount > 0 ? `(${trickCount})` : ""}
      </button>

      {/* Drawer panel */}
      {open && (
        <div
          style={{
            position: "fixed",
            top: 0,
            right: 0,
            bottom: 0,
            width: 280,
            maxWidth: "85vw",
            backgroundColor: "rgba(0, 30, 0, 0.95)",
            borderLeft: "1px solid rgba(255,255,255,0.15)",
            zIndex: 50,
            display: "flex",
            flexDirection: "column",
            animation: "slideInRight 0.2s ease-out",
          }}
        >
          {/* Header */}
          <div style={{
            padding: "12px 16px",
            borderBottom: "1px solid rgba(255,255,255,0.1)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}>
            <span style={{ color: "white", fontWeight: "bold", fontSize: "0.9rem" }}>
              Trick History
            </span>
            <button
              onClick={() => setOpen(false)}
              style={{
                background: "none",
                border: "none",
                color: "rgba(255,255,255,0.6)",
                fontSize: "1.2rem",
                cursor: "pointer",
                padding: "0 4px",
              }}
            >
              ×
            </button>
          </div>

          {/* Trick list */}
          <div style={{ flex: 1, overflowY: "auto", padding: "8px 12px" }}>
            {trickCount === 0 ? (
              <div style={{ color: "rgba(255,255,255,0.4)", textAlign: "center", marginTop: 24, fontSize: "0.85rem" }}>
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
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(0,0,0,0.3)",
            zIndex: 49,
          }}
        />
      )}

      <style>{`
        @keyframes slideInRight {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
      `}</style>
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
    <div style={{
      marginBottom: 8,
      padding: "8px 10px",
      borderRadius: 6,
      backgroundColor: "rgba(255,255,255,0.05)",
      border: "1px solid rgba(255,255,255,0.08)",
    }}>
      {/* Trick header */}
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: 6,
      }}>
        <span style={{ color: "rgba(255,255,255,0.5)", fontSize: "0.7rem", fontWeight: "bold" }}>
          Trick {trickNumber}
        </span>
        {trick.pointsWon > 0 && (
          <span style={{ color: "#ffc107", fontSize: "0.7rem" }}>
            +{trick.pointsWon} pt{trick.pointsWon !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {/* Cards played */}
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
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
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                fontSize: "0.8rem",
                color: isWinner ? "#ffc107" : "rgba(255,255,255,0.75)",
                fontWeight: isWinner ? "bold" : "normal",
              }}
            >
              {isWinner && <span style={{ fontSize: "0.65rem" }}>★</span>}
              <span style={{ width: 55, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: isLocal ? "#7cfc00" : undefined }}>
                {name}
              </span>
              <span style={{ color }}>{rank}{suit}</span>
            </div>
          );
        })}
      </div>

      {/* Winner line */}
      <div style={{ color: "#ffc107", fontSize: "0.7rem", marginTop: 4 }}>
        Won by {trick.winner === localSeat ? "You" : trick.winnerName}
      </div>
    </div>
  );
}
