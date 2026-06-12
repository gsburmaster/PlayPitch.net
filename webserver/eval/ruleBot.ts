/**
 * Faithful TypeScript port of rule_bot.py (the project's eval baseline).
 * Kept byte-for-byte behaviorally equivalent to the Python heuristic so the
 * harness can reproduce the historical "WR vs rule bot" metric. Quirks of the
 * Python bot (e.g. trick-winner ignores the on-jack/off-jack tiebreak) are
 * preserved intentionally.
 */
import { PitchEngine } from "../game/PitchEngine.js";
import { Phase, Suit, Card, cardPoints, OFF_JACK_MAP } from "../game/constants.js";

const SUITS: Suit[] = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES];

function trumpScoreForSuit(hand: Card[], suit: Suit): number {
  const offJack = OFF_JACK_MAP[suit];
  let total = 0;
  for (const card of hand) {
    const isTrump = card.suit === suit;
    const isJoker = card.rank === 11;
    const isOffJack = card.rank === 12 && card.suit === offJack;
    if (isTrump || isJoker || isOffJack) total += cardPoints(card);
  }
  return total;
}

/** Current trick winner by (isValidPlay, rank) — matches rule_bot.py (no jack tiebreak). */
function currentTrickWinner(engine: PitchEngine): { player: number; rank: number } {
  if (engine.currentTrick.length === 0) return { player: -1, rank: -1 };
  let best = engine.currentTrick[0];
  for (let i = 1; i < engine.currentTrick.length; i++) {
    const t = engine.currentTrick[i];
    const tv = engine.isValidPlay(t.card) ? 1 : 0;
    const bv = engine.isValidPlay(best.card) ? 1 : 0;
    if (tv > bv || (tv === bv && t.card.rank > best.card.rank)) best = t;
  }
  return { player: best.seatIndex, rank: best.card.rank };
}

const maxByRank = (idxs: number[], hand: Card[]): number =>
  idxs.reduce((a, b) => (hand[a].rank >= hand[b].rank ? a : b));
const minByRank = (idxs: number[], hand: Card[]): number =>
  idxs.reduce((a, b) => (hand[a].rank <= hand[b].rank ? a : b));

export function pickRuleBotAction(engine: PitchEngine): number {
  const mask = engine.getActionMask();
  const valid: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i] === 1) valid.push(i);
  const cp = engine.currentPlayer;
  const hand = engine.hands[cp];

  if (engine.phase === Phase.BIDDING) {
    let bestScore = 0;
    for (const suit of SUITS) {
      const s = trumpScoreForSuit(hand, suit);
      if (s > bestScore) bestScore = s;
    }
    if (engine.currentPlayer === engine.dealer && engine.currentBid === 0) {
      if (valid.includes(11)) return 11;
    }
    if (bestScore >= 5 && bestScore > engine.currentBid) {
      const action = Math.min(bestScore + 6, 16);
      if (valid.includes(action)) return action;
      for (let a = action; a > 10; a--) if (valid.includes(a)) return a;
    }
    if (valid.includes(10)) return 10;
  } else if (engine.phase === Phase.CHOOSESUIT) {
    let bestScore = -1;
    let bestSuitAction = 19;
    for (const suit of SUITS) {
      const s = trumpScoreForSuit(hand, suit);
      if (s > bestScore) {
        bestScore = s;
        bestSuitAction = 19 + suit;
      }
    }
    if (valid.includes(bestSuitAction)) return bestSuitAction;
    for (let a = 19; a < 23; a++) if (valid.includes(a)) return a;
  } else if (engine.phase === Phase.PLAYING) {
    const playValid = valid.filter((i) => i <= 9);
    if (playValid.length === 0 && valid.includes(23)) return 23;
    const trumpPlays = playValid.filter((i) => engine.isValidPlay(hand[i]));
    const leading = engine.currentTrick.length === 0;

    if (leading) {
      if (trumpPlays.length) return maxByRank(trumpPlays, hand);
      if (playValid.length) return maxByRank(playValid, hand);
    } else {
      const { player: winnerPlayer, rank: winnerRank } = currentTrickWinner(engine);
      const partnerWinning = winnerPlayer !== -1 && winnerPlayer % 2 === cp % 2;
      if (partnerWinning) {
        const nonScorers = playValid.filter((i) => cardPoints(hand[i]) === 0);
        if (nonScorers.length) return minByRank(nonScorers, hand);
        return minByRank(playValid, hand);
      } else {
        const beating = trumpPlays.filter((i) => hand[i].rank > winnerRank);
        if (beating.length) return minByRank(beating, hand);
        const nonScorers = playValid.filter((i) => cardPoints(hand[i]) === 0);
        if (nonScorers.length) return minByRank(nonScorers, hand);
        return minByRank(playValid, hand);
      }
    }
  }

  return valid[0];
}
