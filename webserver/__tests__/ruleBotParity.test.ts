import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";
import { PitchEngine } from "../game/PitchEngine.js";
import { Suit, Phase, Card } from "../game/constants.js";
import { pickRuleBotAction } from "../eval/ruleBot.js";

interface CardRec { s: number | null; r: number; }
interface TrickRec { s: number | null; r: number; seat: number; }
interface Rec {
  seat: number;
  phase: number;
  trumpSuit: number | null;
  currentBid: number;
  currentHighBidder: number;
  dealer: number;
  currentPlayer: number;
  numberOfRoundsPlayed: number;
  scores: number[];
  roundScores: number[];
  playerCardsTaken: number[];
  hands: CardRec[][];
  playedCards: CardRec[];
  currentTrick: TrickRec[];
  ruleBotAction: number;
}

const fixturePath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "fixtures",
  "parity_vectors.json",
);
const records: Rec[] = JSON.parse(readFileSync(fixturePath, "utf-8"));

const toCard = (c: CardRec): Card => ({ suit: c.s as Suit | null, rank: c.r });

function buildEngine(rec: Rec): PitchEngine {
  const e = new PitchEngine(54);
  e.hands = rec.hands.map((h) => h.map(toCard));
  e.playedCards = rec.playedCards.map(toCard);
  e.currentTrick = rec.currentTrick.map((t) => ({
    card: { suit: t.s as Suit | null, rank: t.r },
    seatIndex: t.seat,
  }));
  e.scores = [rec.scores[0], rec.scores[1]];
  e.roundScores = [rec.roundScores[0], rec.roundScores[1]];
  e.playerCardsTaken = [...rec.playerCardsTaken];
  e.currentBid = rec.currentBid;
  e.currentHighBidder = rec.currentHighBidder;
  e.dealer = rec.dealer;
  e.currentPlayer = rec.currentPlayer;
  e.trumpSuit = rec.trumpSuit as Suit | null;
  e.phase = rec.phase as Phase;
  e.numberOfRoundsPlayed = rec.numberOfRoundsPlayed;
  return e;
}

describe("rule-bot parity (TS port vs Python rule_bot.pick_action)", () => {
  it("matches the Python rule bot's action on every golden state", () => {
    let mismatches = 0;
    const examples: string[] = [];
    for (let i = 0; i < records.length; i++) {
      const rec = records[i];
      const got = pickRuleBotAction(buildEngine(rec));
      if (got !== rec.ruleBotAction) {
        mismatches++;
        if (examples.length < 5) {
          examples.push(`record ${i} (phase ${rec.phase}): TS ${got} vs Python ${rec.ruleBotAction}`);
        }
      }
    }
    expect(mismatches, examples.join("; ")).toBe(0);
  });
});
