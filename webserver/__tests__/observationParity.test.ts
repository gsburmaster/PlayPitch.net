import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";
import { PitchEngine } from "../game/PitchEngine.js";
import { Suit, Phase, Card } from "../game/constants.js";
import { flattenObservation } from "../ai/flattenObservation.js";

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
  expected: number[];
}

const fixturePath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "fixtures",
  "parity_vectors.json",
);
const records: Rec[] = JSON.parse(readFileSync(fixturePath, "utf-8"));

const toCard = (c: CardRec): Card => ({ suit: c.s as Suit | null, rank: c.r });

/** Rebuild a PitchEngine whose flattenObservation should match Python exactly. */
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

describe("observation parity (TS flattenObservation vs Python golden vectors)", () => {
  it("loaded a non-trivial fixture covering all phases", () => {
    expect(records.length).toBeGreaterThan(50);
    expect(records[0].expected.length).toBe(141);
    expect(new Set(records.map((r) => r.phase))).toEqual(new Set([0, 1, 2]));
  });

  it("reproduces every Python observation vector within 1e-4", () => {
    let mismatches = 0;
    let firstMismatch = "";
    for (let i = 0; i < records.length; i++) {
      const rec = records[i];
      const got = flattenObservation(buildEngine(rec), rec.seat);
      expect(got.length).toBe(141);
      for (let k = 0; k < 141; k++) {
        if (Math.abs(got[k] - rec.expected[k]) >= 1e-4) {
          mismatches++;
          if (!firstMismatch) {
            firstMismatch = `record ${i} (phase ${rec.phase}) index ${k}: got ${got[k]} expected ${rec.expected[k]}`;
          }
        }
      }
    }
    expect(mismatches, firstMismatch).toBe(0);
  });

  it("matches the crafted on-jack/off-jack trick scenarios (derived feats 4 & 5)", () => {
    // Last two records are the hand-crafted off-jack cases.
    const partnerCase = records[records.length - 2];
    const canWinCase = records[records.length - 1];
    // Derived features occupy the final 10 floats; feat 4 = index 123, feat 5 = 124.
    const F4 = 119 + 4;
    const F5 = 119 + 5;
    // partner (on-jack) wins → partner_winning = 1
    expect(partnerCase.expected[F5]).toBe(1);
    // root holds on-jack vs winning off-jack → can_win = 1
    expect(canWinCase.expected[F4]).toBe(1);
    // And TS reproduces them.
    const p = flattenObservation(buildEngine(partnerCase), partnerCase.seat);
    const c = flattenObservation(buildEngine(canWinCase), canWinCase.seat);
    expect(p[F5]).toBe(1);
    expect(c[F4]).toBe(1);
  });
});
