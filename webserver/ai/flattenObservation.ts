import { PitchEngine } from "../game/PitchEngine.js";
import { Phase } from "../game/constants.js";
import type { Card } from "../game/constants.js";

/**
 * Port of train.py flatten_observation() + pitch_env._get_derived_features().
 * Builds the observation dict from the engine state for a given seat,
 * then flattens arrays and scalars in dict-insertion order into a Float32Array.
 *
 * Observation layout (129 floats):
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
 *   derived_features  (10,) = 10
 *   ─────────────────────────
 *   Total:                  129
 */
export function flattenObservation(engine: PitchEngine, seatIndex: number): Float32Array {
  const out = new Float32Array(129);
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

  // derived_features (10) — mirrors pitch_env._get_derived_features()
  const isPlaying = engine.phase === Phase.PLAYING;
  const cp = seatIndex;
  const myTeam = cp % 2;

  // Collect trump cards in hand
  const trumpCards: Card[] = isPlaying
    ? hand.filter((c) => engine.isValidPlay(c))
    : [];
  const trumpCount = trumpCards.length;

  function cardPoints(card: Card): number {
    if (card.rank === 3) return 3;
    if (card.rank === 15 || card.rank === 12 || card.rank === 10 ||
        card.rank === 11 || card.rank === 2) return 1;
    return 0;
  }

  // feat 0: trump_card_count
  out[offset++] = isPlaying ? trumpCount / 10.0 : 0.0;

  // feat 1: trump_point_count
  const trumpPts = trumpCards.reduce((sum, c) => sum + cardPoints(c), 0);
  out[offset++] = isPlaying ? trumpPts / 7.0 : 0.0;

  // feat 2: void_in_trump
  out[offset++] = isPlaying ? (trumpCount === 0 ? 1.0 : 0.0) : 0.0;

  // feat 3: highest_trump_rank
  if (isPlaying && trumpCount > 0) {
    const maxRank = Math.max(...trumpCards.map((c) => c.rank));
    out[offset++] = (maxRank - 2) / 13.0;
  } else {
    out[offset++] = 0.0;
  }

  // feat 4: can_win_trick
  if (isPlaying) {
    if (engine.currentTrick.length === 0) {
      out[offset++] = 1.0; // leading
    } else {
      // Find current trick winner rank
      let winnerEntry = engine.currentTrick[0];
      for (let i = 1; i < engine.currentTrick.length; i++) {
        const challenger = engine.currentTrick[i];
        const challValid = engine.isValidPlay(challenger.card);
        const curValid = engine.isValidPlay(winnerEntry.card);
        if ((challValid && !curValid) ||
            (challValid && curValid && challenger.card.rank > winnerEntry.card.rank)) {
          winnerEntry = challenger;
        }
      }
      const winnerRank = winnerEntry.card.rank;
      const canBeat = trumpCards.some((c) => c.rank > winnerRank);
      out[offset++] = canBeat ? 1.0 : 0.0;
    }
  } else {
    out[offset++] = 0.0;
  }

  // feat 5: partner_winning
  if (isPlaying && engine.currentTrick.length > 0) {
    let winnerEntry = engine.currentTrick[0];
    for (let i = 1; i < engine.currentTrick.length; i++) {
      const challenger = engine.currentTrick[i];
      const challValid = engine.isValidPlay(challenger.card);
      const curValid = engine.isValidPlay(winnerEntry.card);
      if ((challValid && !curValid) ||
          (challValid && curValid && challenger.card.rank > winnerEntry.card.rank)) {
        winnerEntry = challenger;
      }
    }
    out[offset++] = (winnerEntry.seatIndex % 2 === myTeam) ? 1.0 : 0.0;
  } else {
    out[offset++] = 0.0;
  }

  // feat 6: am_high_bidder (always)
  out[offset++] = engine.currentHighBidder === cp ? 1.0 : 0.0;

  // feat 7: bid_deficit (always) — max bid is 12 (double moon)
  const myRoundScore = engine.roundScores[myTeam];
  out[offset++] = Math.max(0, engine.currentBid - myRoundScore) / 12.0;

  // feat 8: tricks_remaining (always) — max hand size is 9 at deal, 6 after fill
  const maxHandSize = Math.max(...engine.hands.map((h) => h.length));
  out[offset++] = maxHandSize / 9.0;

  // feat 9: point_cards_remaining (always)
  const totalScored = engine.roundScores[0] + engine.roundScores[1];
  out[offset++] = 1.0 - Math.min(totalScored, 9) / 9.0;

  return out;
}
