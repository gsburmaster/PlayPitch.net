import { describe, it, expect } from "vitest";
import { PitchEngine } from "../game/PitchEngine.js";
import { Suit, Phase, Card } from "../game/constants.js";
import { pickRuleBotAction } from "../eval/ruleBot.js";
import {
  determinize,
  consistentDeterminize,
  trumpCountPosterior,
  voidSeats,
  ismctsSearch,
  onePlyLookahead,
  type Evaluator,
} from "../ai/search.js";

// Deterministic PRNG so determinization/search tests are reproducible.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const card = (suit: Suit | null, rank: number): Card => ({ suit, rank });
const sortKey = (c: Card) => `${c.suit}:${c.rank}`;
const multiset = (cards: Card[]) => cards.map(sortKey).sort();

/** Minimal hand-built PLAYING position with a leftover deck (Hearts trump). */
function makePlayingEngine(): PitchEngine {
  const e = new PitchEngine(54);
  e.reset(0);
  e.phase = Phase.PLAYING;
  e.trumpSuit = Suit.HEARTS;
  e.currentBid = 4;
  e.currentHighBidder = 0;
  e.dealer = 3;
  e.currentPlayer = 0;
  e.playingIterator = 0;
  e.currentTrick = [];
  e.tricks = [];
  e.playedCards = [];
  e.roundScores = [0, 0];
  e.scores = [0, 0];
  e.playerCardsTaken = [0, 0, 0, 0];
  e.hands = [
    [card(Suit.HEARTS, 15), card(Suit.HEARTS, 5)], // root (seat 0): A♥, 5♥ — both trump
    [card(Suit.HEARTS, 6), card(Suit.CLUBS, 7)],
    [card(Suit.HEARTS, 4), card(Suit.CLUBS, 8)],
    [card(Suit.HEARTS, 9), card(Suit.CLUBS, 10)],
  ];
  e.deck = [card(Suit.DIAMONDS, 13), card(Suit.SPADES, 14), card(Suit.HEARTS, 12)];
  return e;
}

// A uniform evaluator: no preference, zero value. Exercises the machinery.
const uniformEval: Evaluator = async () => ({ policy: new Array(24).fill(0), value: 0 });

describe("PitchEngine.clone", () => {
  it("produces an independent deep copy", () => {
    const e = makePlayingEngine();
    const c = e.clone();
    expect(c.hands).toEqual(e.hands);
    expect(c.deck).toEqual(e.deck);
    expect(c.scores).toEqual(e.scores);

    // Mutating the clone must not touch the original.
    c.hands[0].push(card(Suit.SPADES, 2));
    c.hands[1][0].rank = 99;
    c.scores[0] = 50;
    c.trumpSuit = Suit.SPADES;
    c.deck.pop();

    expect(e.hands[0]).toHaveLength(2);
    expect(e.hands[1][0].rank).toBe(6);
    expect(e.scores[0]).toBe(0);
    expect(e.trumpSuit).toBe(Suit.HEARTS);
    expect(e.deck).toHaveLength(3);
  });

  it("conserves all 54 cards from a fresh deal", () => {
    const e = new PitchEngine(54);
    e.reset(0);
    const c = e.clone();
    const total =
      c.hands.flat().length + c.deck.length + c.playedCards.length;
    expect(total).toBe(54);
    expect(c.hands.flat().length).toBe(36);
    expect(c.deck.length).toBe(18);
  });
});

describe("determinize", () => {
  it("keeps the root hand fixed and conserves the hidden-card multiset", () => {
    const e = makePlayingEngine();
    const seat1Before = e.hands[1].map((c) => ({ ...c }));
    const poolBefore = multiset([
      ...e.hands[1],
      ...e.hands[2],
      ...e.hands[3],
      ...e.deck,
    ]);

    const sim = determinize(e, 0, mulberry32(123));

    // Root hand untouched (same cards, same order).
    expect(sim.hands[0]).toEqual(e.hands[0]);
    // Hand sizes and deck size preserved.
    expect(sim.hands[1].length).toBe(2);
    expect(sim.hands[2].length).toBe(2);
    expect(sim.hands[3].length).toBe(2);
    expect(sim.deck.length).toBe(3);
    // Hidden-card multiset conserved (no cards invented or lost).
    const poolAfter = multiset([
      ...sim.hands[1],
      ...sim.hands[2],
      ...sim.hands[3],
      ...sim.deck,
    ]);
    expect(poolAfter).toEqual(poolBefore);
    // Original engine untouched.
    expect(e.hands[1]).toEqual(seat1Before);
  });

  it("actually reshuffles hidden cards (not an identity)", () => {
    const e = makePlayingEngine();
    // Give the hidden region enough cards that a permutation is near-certain.
    e.hands[1] = [card(Suit.HEARTS, 6), card(Suit.CLUBS, 7), card(Suit.DIAMONDS, 5)];
    e.hands[2] = [card(Suit.HEARTS, 4), card(Suit.CLUBS, 8), card(Suit.SPADES, 9)];
    e.hands[3] = [card(Suit.HEARTS, 9), card(Suit.CLUBS, 10), card(Suit.SPADES, 12)];
    const before = JSON.stringify([e.hands[1], e.hands[2], e.hands[3], e.deck]);
    const sim = determinize(e, 0, mulberry32(7));
    const after = JSON.stringify([sim.hands[1], sim.hands[2], sim.hands[3], sim.deck]);
    expect(after).not.toBe(before);
  });
});

describe("ismctsSearch", () => {
  it("returns a valid (legal) action", async () => {
    const e = makePlayingEngine();
    const action = await ismctsSearch(e, 0, {
      evaluate: uniformEval,
      rng: mulberry32(1),
      maxSims: 40,
      budgetMs: 100000,
    });
    const mask = e.getActionMask();
    expect(mask[action]).toBe(1);
  });

  it("is reproducible given a fixed rng, evaluator and sim count", async () => {
    const opts = () => ({
      evaluate: uniformEval,
      rng: mulberry32(99),
      maxSims: 60,
      budgetMs: 100000,
    });
    const a1 = await ismctsSearch(makePlayingEngine(), 0, opts());
    const a2 = await ismctsSearch(makePlayingEngine(), 0, opts());
    expect(a1).toBe(a2);
  });

  it("does not mutate the caller's engine", async () => {
    const e = makePlayingEngine();
    const snapshot = JSON.stringify({ hands: e.hands, deck: e.deck, scores: e.scores });
    await ismctsSearch(e, 0, { evaluate: uniformEval, rng: mulberry32(3), maxSims: 30, budgetMs: 100000 });
    expect(JSON.stringify({ hands: e.hands, deck: e.deck, scores: e.scores })).toBe(snapshot);
  });
});

describe("onePlyLookahead", () => {
  it("prefers the line the value function rates higher", async () => {
    const e = makePlayingEngine();
    // Value the position highly iff the root's Ace of trump has been played.
    const aceSeekingEval: Evaluator = async (eng) => {
      const acePlayed = eng.playedCards.some(
        (c) => c.suit === Suit.HEARTS && c.rank === 15,
      );
      return { policy: new Array(24).fill(0), value: acePlayed ? 5 : -5 };
    };
    const action = await onePlyLookahead(e, 0, { evaluate: aceSeekingEval });
    // Action 0 is A♥ in the root hand; the evaluator should steer to it.
    expect(e.hands[0][action]).toEqual(card(Suit.HEARTS, 15));
  });

  it("returns a valid action under a uniform evaluator", async () => {
    const e = makePlayingEngine();
    const action = await onePlyLookahead(e, 0, { evaluate: uniformEval });
    expect(e.getActionMask()[action]).toBe(1);
  });
});

/** PLAYING-phase states from rule-bot games (real discard/fill history). */
function realPlayingStates(nGames: number, every = 3): PitchEngine[] {
  const out: PitchEngine[] = [];
  for (let g = 0; g < nGames; g++) {
    const e = new PitchEngine(54);
    e.reset(g % 4);
    for (let step = 0; step < 3000 && !e.gameOver; step++) {
      if (e.phase === Phase.PLAYING && step % every === 0) out.push(e.clone());
      e.step(pickRuleBotAction(e));
    }
  }
  return out;
}

const trumpsHeld = (e: PitchEngine, p: number) => e.hands[p].filter((c) => e.isValidPlay(c)).length;

describe("consistentDeterminize", () => {
  const states = realPlayingStates(15);

  it("preserves the root hand, sizes and the hidden-card multiset", () => {
    const rng = mulberry32(1);
    for (const e of states) {
      const root = e.currentPlayer;
      const others = [0, 1, 2, 3].filter((p) => p !== root);
      const sim = consistentDeterminize(e, root, rng);
      expect(sim.hands[root]).toEqual(e.hands[root]);
      for (let p = 0; p < 4; p++) expect(sim.hands[p]).toHaveLength(e.hands[p].length);
      expect(sim.deck).toHaveLength(e.deck.length);
      expect(multiset([...others.flatMap((p) => sim.hands[p]), ...sim.deck]))
        .toEqual(multiset([...others.flatMap((p) => e.hands[p]), ...e.deck]));
    }
  });

  it("always has the true trump counts in the posterior's support", () => {
    for (const e of states) {
      const root = e.currentPlayer;
      const others = [0, 1, 2, 3].filter((p) => p !== root);
      const unseen = [...others.flatMap((p) => e.hands[p]), ...e.deck].filter((c) => e.isValidPlay(c));
      const post = trumpCountPosterior(e, root, unseen.length);
      expect(post).not.toBeNull();
      expect(post!.probs.reduce((s, x) => s + x, 0)).toBeCloseTo(1, 9);
      const truth = others.map((p) => trumpsHeld(e, p));
      expect(post!.outcomes.some((o) => others.every((p, i) => o[p] === truth[i]))).toBe(true);
    }
  });

  it("never gives a shown-void seat a trump", () => {
    const rng = mulberry32(2);
    let checked = 0;
    for (const e of realPlayingStates(15, 1)) {
      const root = e.currentPlayer;
      const cache = {};
      for (const p of voidSeats(e)) {
        if (p === root) continue;
        for (let k = 0; k < 3; k++) expect(trumpsHeld(consistentDeterminize(e, root, rng, cache), p)).toBe(0);
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(0);
  });

  it("guesses hidden trump counts much better than a uniform re-deal", () => {
    const rng = mulberry32(3);
    let errU = 0;
    let errC = 0;
    for (const e of states) {
      const root = e.currentPlayer;
      const cache = {};
      for (let k = 0; k < 5; k++) {
        const u = determinize(e, root, rng);
        const c = consistentDeterminize(e, root, rng, cache);
        for (let p = 0; p < 4; p++) {
          if (p === root) continue;
          errU += Math.abs(trumpsHeld(u, p) - trumpsHeld(e, p));
          errC += Math.abs(trumpsHeld(c, p) - trumpsHeld(e, p));
        }
      }
    }
    expect(errC).toBeLessThan(0.6 * errU);
  });
});
