import { PitchEngine } from "../game/PitchEngine.js";
import type { Card } from "../game/constants.js";

/**
 * Port of multi_agent.py flatten_observation().
 * Builds the observation dict from the engine state for a given seat,
 * then flattens arrays and scalars in dict-insertion order into a Float32Array.
 *
 * Observation layout (119 floats):
 *   hand          (10,2) = 20   — current player's hand, suit(0-4)+rank
 *   played_cards  (24,2) = 48   — all cards played this round
 *   scores        (2,)   = 2    — team scores
 *   round_scores  (2,)   = 2    — round scores
 *   current_trick (4,3)  = 12   — (suit, rank, seatIndex) per trick card
 *   current_bid          = 1
 *   current_high_bidder  = 1
 *   dealer               = 1
 *   current_player       = 1
 *   trump_suit           = 1    — 0-3 or 4 if no trump
 *   phase                = 1
 *   number_of_rounds_played = 1
 *   player_cards_taken (4,) = 4
 *   action_mask       (24,) = 24
 *   ─────────────────────────
 *   Total:                  119
 */
export function flattenObservation(engine: PitchEngine, seatIndex: number): Float32Array {
  const out = new Float32Array(119);
  let offset = 0;

  // helper to write a card (suit, rank) pair
  function writeCard(card: Card | null) {
    if (card) {
      out[offset++] = card.suit !== null ? card.suit : 4;
      out[offset++] = card.rank;
    } else {
      out[offset++] = 0;
      out[offset++] = 0;
    }
  }

  // hand — pad to 10 cards
  const hand = engine.hands[seatIndex];
  for (let i = 0; i < 10; i++) {
    writeCard(i < hand.length ? hand[i] : null);
  }

  // played_cards — pad to 24
  for (let i = 0; i < 24; i++) {
    writeCard(i < engine.playedCards.length ? engine.playedCards[i] : null);
  }

  // scores (2)
  out[offset++] = engine.scores[0];
  out[offset++] = engine.scores[1];

  // round_scores (2)
  out[offset++] = engine.roundScores[0];
  out[offset++] = engine.roundScores[1];

  // current_trick — pad to 4 entries, each (suit, rank, seatIndex)
  for (let i = 0; i < 4; i++) {
    if (i < engine.currentTrick.length) {
      const entry = engine.currentTrick[i];
      out[offset++] = entry.card.suit !== null ? entry.card.suit : 4;
      out[offset++] = entry.card.rank;
      out[offset++] = entry.seatIndex;
    } else {
      out[offset++] = 0;
      out[offset++] = 0;
      out[offset++] = 0;
    }
  }

  // scalars
  out[offset++] = engine.currentBid;
  out[offset++] = engine.currentHighBidder;
  out[offset++] = engine.dealer;
  out[offset++] = engine.currentPlayer;
  out[offset++] = engine.trumpSuit !== null ? engine.trumpSuit : 4;
  out[offset++] = engine.phase;
  out[offset++] = engine.numberOfRoundsPlayed;

  // player_cards_taken (4)
  for (let i = 0; i < 4; i++) {
    out[offset++] = engine.playerCardsTaken[i];
  }

  // action_mask (24)
  const mask = engine.getActionMask();
  for (let i = 0; i < 24; i++) {
    out[offset++] = mask[i];
  }

  return out;
}
