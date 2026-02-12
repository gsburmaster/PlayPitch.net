import { PitchEngine } from "../game/PitchEngine.js";
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
export declare function flattenObservation(engine: PitchEngine, seatIndex: number): Float32Array;
