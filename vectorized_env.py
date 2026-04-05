"""
GPU-native vectorized Pitch environment.

Runs N games simultaneously using PyTorch tensors. All game state lives on
a single device (CPU/MPS/CUDA), and all operations are batched tensor ops
— no Python loops over individual games in the hot path.

Card encoding: suit * 16 + rank  (0 = empty slot)
  Hearts=0, Diamonds=1, Clubs=2, Spades=3, Joker suit=4
  Ranks 2-15 (J=12, Q=13, K=14, A=15), Joker rank=11
  So Joker = 4*16 + 11 = 75, empty = 0
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Card encoding helpers
# ---------------------------------------------------------------------------

JOKER_CODE = 75  # 4 * 16 + 11

# Pre-built template deck: 4 suits × (2-10, 12-15) + 2 jokers = 54 cards
# Matches PitchEnv._create_deck() order: Hearts 2-10,12-15, Diamonds ..., + jokers
_TEMPLATE_CARDS = []
for _suit in range(4):
    for _rank in range(2, 11):  # 2-10
        _TEMPLATE_CARDS.append(_suit * 16 + _rank)
    for _rank in range(12, 16):  # J,Q,K,A
        _TEMPLATE_CARDS.append(_suit * 16 + _rank)
_TEMPLATE_CARDS.extend([JOKER_CODE, JOKER_CODE])
TEMPLATE_DECK = torch.tensor(_TEMPLATE_CARDS, dtype=torch.int8)
assert len(TEMPLATE_DECK) == 54


def card_suit(c: torch.Tensor) -> torch.Tensor:
    """Extract suit from encoded card(s). Returns 0-4 (4=joker)."""
    return c.to(torch.int16) // 16


def card_rank(c: torch.Tensor) -> torch.Tensor:
    """Extract rank from encoded card(s). Returns 2-15 (11=joker)."""
    return c.to(torch.int16) % 16


def encode_card(suit: int, rank: int) -> int:
    """Encode a single (suit, rank) pair. Use suit=4 for joker."""
    return suit * 16 + rank


# Points lookup: indexed by rank (0-15). Ranks with points: 2→1, 3→3, 10→1, 11(joker)→1, 12(J)→1, 15(A)→1
_POINTS_BY_RANK = [0] * 16
_POINTS_BY_RANK[2] = 1
_POINTS_BY_RANK[3] = 3
_POINTS_BY_RANK[10] = 1
_POINTS_BY_RANK[11] = 1   # joker
_POINTS_BY_RANK[12] = 1   # jack
_POINTS_BY_RANK[15] = 1   # ace
POINTS_TABLE = torch.tensor(_POINTS_BY_RANK, dtype=torch.int8)


# Phase constants
PHASE_BIDDING = 0
PHASE_CHOOSESUIT = 1
PHASE_PLAYING = 2


class VectorizedPitchEnv:
    """N simultaneous Pitch games as GPU tensors."""

    def __init__(self, N: int, device: torch.device, win_threshold: int = 54):
        self.N = N
        self.device = device
        self.win_threshold = win_threshold

        # Points table on device
        self.points_table = POINTS_TABLE.to(device)

        # --- Card storage ---
        # hands[game, player, slot] = card code (0 = empty)
        self.hands = torch.zeros(N, 4, 10, dtype=torch.int8, device=device)
        # deck[game, position] = card code
        self.deck = torch.zeros(N, 54, dtype=torch.int8, device=device)
        # deck_top[game] = next card index to deal from
        self.deck_top = torch.zeros(N, dtype=torch.int32, device=device)

        # --- Current trick ---
        self.trick_cards = torch.zeros(N, 4, dtype=torch.int8, device=device)
        self.trick_players = torch.full((N, 4), -1, dtype=torch.int8, device=device)

        # --- Played cards history (for observations) ---
        self.played_cards = torch.zeros(N, 24, dtype=torch.int8, device=device)
        self.played_cards_idx = torch.zeros(N, dtype=torch.int32, device=device)

        # --- Scalar game state ---
        self.phase = torch.zeros(N, dtype=torch.int8, device=device)
        self.current_player = torch.zeros(N, dtype=torch.int8, device=device)
        self.dealer = torch.zeros(N, dtype=torch.int8, device=device)
        self.trump_suit = torch.full((N,), -1, dtype=torch.int8, device=device)
        self.current_bid = torch.zeros(N, dtype=torch.int8, device=device)
        self.current_high_bidder = torch.zeros(N, dtype=torch.int8, device=device)
        self.playing_iterator = torch.zeros(N, dtype=torch.int8, device=device)

        # --- Scores ---
        self.scores = torch.zeros(N, 2, dtype=torch.int16, device=device)
        self.round_scores = torch.zeros(N, 2, dtype=torch.int8, device=device)
        self.last_trick_points = torch.zeros(N, 2, dtype=torch.int8, device=device)
        self.scores_before = torch.zeros(N, 2, dtype=torch.int16, device=device)
        self._saved_round_scores = torch.zeros(N, 2, dtype=torch.int8, device=device)

        # --- Misc ---
        self.player_cards_taken = torch.full((N, 4), -1, dtype=torch.int8, device=device)
        self.num_rounds_played = torch.zeros(N, dtype=torch.int32, device=device)
        self.done = torch.ones(N, dtype=torch.bool, device=device)  # start as done → need reset
        self.trick_winner = torch.zeros(N, dtype=torch.int8, device=device)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_all(self):
        """Reset all N games. Returns initial observations (N, obs_dim)."""
        mask = torch.ones(self.N, dtype=torch.bool, device=self.device)
        self._reset_games(mask)
        obs, _ = self.get_observations()
        return obs

    def reset_done(self):
        """Reset only games that are done."""
        done_mask = self.done.clone()
        if done_mask.any():
            self._reset_games(done_mask)

    def _reset_games(self, mask: torch.Tensor):
        """Reset games indicated by boolean mask."""
        n = mask.sum().item()
        if n == 0:
            return

        # Clear state
        self.hands[mask] = 0
        self.trick_cards[mask] = 0
        self.trick_players[mask] = -1
        self.played_cards[mask] = 0
        self.played_cards_idx[mask] = 0
        self.phase[mask] = PHASE_BIDDING
        self.trump_suit[mask] = -1
        self.current_bid[mask] = 0
        self.current_high_bidder[mask] = 0
        self.playing_iterator[mask] = 0
        self.round_scores[mask] = 0
        self.last_trick_points[mask] = 0
        self.scores_before[mask] = 0
        self._saved_round_scores[mask] = 0
        self.player_cards_taken[mask] = -1
        self.trick_winner[mask] = 0
        self.done[mask] = False

        # Scores and rounds persist across rounds but reset on game reset
        self.scores[mask] = 0
        self.num_rounds_played[mask] = 0

        # Random dealer for each game
        dealers = torch.randint(0, 4, (n,), device=self.device, dtype=torch.int8)
        self.dealer[mask] = dealers
        self.current_player[mask] = (dealers + 1) % 4

        # Create shuffled decks and deal
        self._create_and_shuffle_decks(mask)
        self._deal_cards(mask)

    def _reset_round(self, mask: torch.Tensor):
        """Reset for a new round (not a new game) — keeps scores, advances dealer."""
        n = mask.sum().item()
        if n == 0:
            return

        self.hands[mask] = 0
        self.trick_cards[mask] = 0
        self.trick_players[mask] = -1
        self.played_cards[mask] = 0
        self.played_cards_idx[mask] = 0
        self.phase[mask] = PHASE_BIDDING
        self.trump_suit[mask] = -1
        self.current_bid[mask] = 0
        self.current_high_bidder[mask] = 0
        self.playing_iterator[mask] = 0
        self.round_scores[mask] = 0
        self.last_trick_points[mask] = 0
        self.player_cards_taken[mask] = -1
        self.trick_winner[mask] = 0

        # Advance dealer
        self.dealer[mask] = (self.dealer[mask] + 1) % 4
        self.current_player[mask] = (self.dealer[mask] + 1) % 4
        self.num_rounds_played[mask] += 1

        self._create_and_shuffle_decks(mask)
        self._deal_cards(mask)

    # ------------------------------------------------------------------
    # Deck creation and dealing
    # ------------------------------------------------------------------

    def _create_and_shuffle_decks(self, mask: torch.Tensor):
        """Create and shuffle decks for masked games."""
        n = mask.sum().item()
        if n == 0:
            return

        template = TEMPLATE_DECK.to(self.device)
        decks = template.unsqueeze(0).expand(n, -1).clone()

        # Shuffle via argsort(rand)
        shuffle_idx = torch.argsort(
            torch.rand(n, 54, device=self.device), dim=1
        )
        decks = decks.gather(1, shuffle_idx.to(torch.int64)).to(torch.int8)

        self.deck[mask] = decks
        self.deck_top[mask] = 0

    def _deal_cards(self, mask: torch.Tensor):
        """Deal 9 cards to each player from their deck.

        Matches PitchEnv._deal_cards(): 9 rounds, each round deals to
        players 0,1,2,3 by popping from deck end.
        """
        n = mask.sum().item()
        if n == 0:
            return

        # Deck has 54 cards. Deal from the end (index 53 down to 18).
        # Round r, player p gets deck[53 - (r*4 + p)]
        # After dealing: 36 cards dealt, deck_top stays 0, "live" deck is indices 0-17
        deal_indices = []
        for r in range(9):
            for p in range(4):
                deal_indices.append(53 - (r * 4 + p))
        # deal_indices[i] gives the deck index for the i-th dealt card
        # Card i goes to player (i % 4), hand slot (i // 4)

        deal_idx_t = torch.tensor(deal_indices, dtype=torch.int64, device=self.device)
        dealt = self.deck[mask][:, deal_idx_t.to(torch.int64)]  # (n, 36)

        # Reshape: 36 cards → 9 rounds × 4 players
        dealt = dealt.reshape(n, 9, 4)  # (n, round, player)
        dealt = dealt.permute(0, 2, 1)  # (n, player, round) = (n, 4, 9)

        self.hands[mask, :, :9] = dealt.to(torch.int8)
        self.hands[mask, :, 9] = 0  # slot 10 empty

        # Mark dealt positions as 0 in deck, set deck_top to 0
        # The "remaining" deck is indices 0..17
        for idx in deal_indices:
            self.deck[mask, idx] = 0
        self.deck_top[mask] = 0

    # ------------------------------------------------------------------
    # Card validity and points
    # ------------------------------------------------------------------

    def _is_valid_play(self, card_codes: torch.Tensor) -> torch.Tensor:
        """Check if cards are valid plays (trump, joker, or off-jack).

        card_codes: any shape tensor of encoded cards
        Returns: bool tensor of same shape
        """
        suits = card_suit(card_codes)
        ranks = card_rank(card_codes)

        # Expand trump_suit to match card_codes shape
        # card_codes could be (N, 4, 10) or (N, 10) etc.
        # We need trump_suit (N,) to broadcast
        extra_dims = card_codes.dim() - 1
        trump = self.trump_suit.view(self.N, *([1] * extra_dims))

        is_trump = suits == trump
        is_joker = card_codes == JOKER_CODE
        # Off-jack: jack (rank 12) of the same-color suit
        # Hearts(0)↔Diamonds(1), Clubs(2)↔Spades(3) — XOR with 1
        off_jack_suit = trump ^ 1
        is_off_jack = (ranks == 12) & (suits == off_jack_suit)

        is_card = card_codes != 0
        return is_card & (is_trump | is_joker | is_off_jack)

    def _card_points_lookup(self, card_codes: torch.Tensor) -> torch.Tensor:
        """Look up points for encoded cards. Returns int8 tensor of same shape."""
        ranks = card_rank(card_codes)
        # Clamp to valid index range (empty card has rank 0 → 0 points)
        ranks = ranks.clamp(0, 15).long()
        pts = self.points_table[ranks]
        # Empty cards get 0
        pts = pts * (card_codes != 0).to(torch.int8)
        return pts

    # ------------------------------------------------------------------
    # Action mask
    # ------------------------------------------------------------------

    def _get_action_mask(self) -> torch.Tensor:
        """Compute (N, 24) action mask for all games."""
        masks = torch.zeros(self.N, 24, dtype=torch.int8, device=self.device)
        one = torch.tensor(1, dtype=torch.int8, device=self.device)

        # --- Bidding phase ---
        bidding = self.phase == PHASE_BIDDING
        if bidding.any():
            cur_bid_mask = self.current_bid + 6  # bid value as mask index
            # Pass: always valid unless dealer with no bids
            can_pass = bidding & (
                (self.current_bid > 0) | (self.current_player != self.dealer)
            )
            masks[:, 10] = torch.where(can_pass, one, masks[:, 10])

            # Bids above current bid (actions 11-17)
            for action in range(11, 18):
                can_bid = bidding & (cur_bid_mask < action)
                masks[:, action] = torch.where(can_bid, one, masks[:, action])

            # Double moon (action 18): dealer only when someone bid moon (bid=11 → mask=17)
            can_dbl = bidding & (self.current_player == self.dealer) & (cur_bid_mask == 17)
            masks[:, 18] = torch.where(can_dbl, one, masks[:, 18])

        # --- Choose suit phase ---
        choosing = (self.phase == PHASE_CHOOSESUIT) & (
            self.current_player == self.current_high_bidder
        )
        if choosing.any():
            masks[:, 19:23] = torch.where(
                choosing.unsqueeze(1), one, masks[:, 19:23]
            )

        # --- Playing phase ---
        playing = self.phase == PHASE_PLAYING
        if playing.any():
            game_idx = torch.arange(self.N, device=self.device)
            cp = self.current_player.long()
            hand = self.hands[game_idx, cp]  # (N, 10)
            valid = self._is_valid_play(hand) & (hand != 0)  # (N, 10)
            any_valid = valid.any(dim=1)  # (N,)

            # Card slots 0-9
            masks[:, :10] = torch.where(
                (playing & any_valid).unsqueeze(1),
                valid.to(torch.int8),
                masks[:, :10],
            )
            # No valid play (action 23)
            masks[:, 23] = torch.where(
                playing & ~any_valid, one, masks[:, 23]
            )

        return masks

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def get_observations(self) -> torch.Tensor:
        """Build (N, obs_dim) float tensor matching flatten_observation() layout.

        Layout (matching PitchEnv._get_observation dict iteration order):
          hand:              (10, 2) = 20 — suit, rank per card
          played_cards:      (24, 2) = 48 — suit, rank per card
          scores:            (2,)    = 2
          round_scores:      (2,)    = 2
          current_trick:     (4, 3)  = 12 — suit, rank, player per entry
          current_bid:       1
          current_high_bidder: 1
          dealer:            1
          current_player:    1
          trump_suit:        1  (0-3, 4 for no trump)
          phase:             1
          number_of_rounds_played: 1
          player_cards_taken: (4,) = 4
          action_mask:       (24,) = 24
          derived_features:  (10,) = 10
          Total: 20 + 48 + 2 + 2 + 12 + 7 + 4 + 24 + 10 = 129
        """
        game_idx = torch.arange(self.N, device=self.device)
        cp = self.current_player.long()

        # Hand: current player's hand → (N, 10, 2) → (N, 20)
        hand = self.hands[game_idx, cp]  # (N, 10) card codes
        hand_suit = card_suit(hand).float()  # joker suit=4, empty=0
        hand_rank = card_rank(hand).float()  # empty=0
        # Fix: empty slots should be (0,0), joker should be (4, 11)
        is_empty = (hand == 0)
        hand_suit = torch.where(is_empty, torch.zeros_like(hand_suit), hand_suit)
        hand_rank = torch.where(is_empty, torch.zeros_like(hand_rank), hand_rank)
        hand_flat = torch.stack([hand_suit, hand_rank], dim=2).reshape(self.N, 20)

        # Played cards: (N, 24) → (N, 24, 2) → (N, 48)
        pc = self.played_cards  # (N, 24) card codes
        pc_suit = card_suit(pc).float()
        pc_rank = card_rank(pc).float()
        pc_empty = (pc == 0)
        pc_suit = torch.where(pc_empty, torch.zeros_like(pc_suit), pc_suit)
        pc_rank = torch.where(pc_empty, torch.zeros_like(pc_rank), pc_rank)
        pc_flat = torch.stack([pc_suit, pc_rank], dim=2).reshape(self.N, 48)

        # Scores: (N, 2)
        scores_flat = self.scores.float()  # (N, 2)

        # Round scores: (N, 2)
        round_scores_flat = self.round_scores.float()  # (N, 2)

        # Current trick: (N, 4, 3) → (N, 12)
        tc = self.trick_cards  # (N, 4)
        tp = self.trick_players  # (N, 4)
        tc_suit = card_suit(tc).float()
        tc_rank = card_rank(tc).float()
        tc_empty = (tc == 0)
        tc_suit = torch.where(tc_empty, torch.zeros_like(tc_suit), tc_suit)
        tc_rank = torch.where(tc_empty, torch.zeros_like(tc_rank), tc_rank)
        tp_float = tp.float()
        tp_float = torch.where(tc_empty, torch.zeros_like(tp_float), tp_float)
        trick_flat = torch.stack([tc_suit, tc_rank, tp_float], dim=2).reshape(self.N, 12)

        # Scalars
        # trump_suit: -1 (no trump) → 4 to match Python env
        trump_obs = self.trump_suit.clone()
        trump_obs[trump_obs < 0] = 4
        scalars = torch.stack([
            self.current_bid.float(),
            self.current_high_bidder.float(),
            self.dealer.float(),
            self.current_player.float(),
            trump_obs.float(),
            self.phase.float(),
            self.num_rounds_played.float(),
        ], dim=1)  # (N, 7)

        # Player cards taken: (N, 4)
        pct_flat = self.player_cards_taken.float()

        # Action mask: (N, 24)
        mask = self._get_action_mask()
        mask_flat = mask.float()

        # --- Derived features (10 floats, all [0,1]) ---
        is_playing = (self.phase == PHASE_PLAYING).float()  # (N,)

        # Current player's hand cards
        hand_codes = self.hands[game_idx, cp]  # (N, 10)
        h_suit = card_suit(hand_codes)         # (N, 10) int16
        h_rank = card_rank(hand_codes)         # (N, 10) int16
        is_empty_hand = (hand_codes == 0)

        # Trump suit per game: -1 means no trump; clamp to 0 for comparisons
        ts = self.trump_suit.long()  # (N,)

        # off-jack suit: CLUBS↔SPADES (0↔3), DIAMONDS↔HEARTS (1↔2)
        off_jack_map = torch.tensor([3, 2, 1, 0], dtype=torch.long, device=self.device)
        off_jack_suit = torch.where(ts >= 0, off_jack_map[ts.clamp(min=0)],
                                    torch.full_like(ts, -1))  # (N,)

        # is_trump_card: suit == trump OR rank == 11 (joker) OR off-jack
        h_suit_long = h_suit.long()  # (N, 10)
        h_rank_long = h_rank.long()  # (N, 10)
        ts_exp = ts.unsqueeze(1).expand_as(h_suit_long)   # (N, 10)
        oj_exp = off_jack_suit.unsqueeze(1).expand_as(h_suit_long)  # (N, 10)

        is_trump_card = (
            (~is_empty_hand) & (
                (h_suit_long == ts_exp) |
                (h_rank_long == 11) |
                ((h_rank_long == 12) & (h_suit_long == oj_exp))
            )
        ).float()  # (N, 10)

        # Points by rank for hand cards
        points_table = self.points_table.long()  # (16,)
        h_rank_clamped = h_rank_long.clamp(0, 15)
        h_pts = points_table[h_rank_clamped].float()  # (N, 10)

        trump_count = is_trump_card.sum(dim=1)                    # (N,)
        trump_pts = (is_trump_card * h_pts).sum(dim=1)            # (N,)

        # feat 0: trump_card_count
        feat0 = trump_count / 10.0

        # feat 1: trump_point_count
        feat1 = trump_pts / 7.0

        # feat 2: void_in_trump
        feat2 = (trump_count == 0).float()

        # feat 3: highest_trump_rank
        trump_ranks = torch.where(is_trump_card.bool(), h_rank.float(),
                                  torch.zeros_like(h_rank.float()))
        max_trump_rank = trump_ranks.max(dim=1).values  # (N,)
        feat3 = (max_trump_rank - 2.0).clamp(min=0) / 13.0

        # feat 4: can_win_trick (1 if leading OR any trump rank > current winner rank)
        # Current winner rank: max over trick slots with valid-play weighting
        tc_codes = self.trick_cards   # (N, 4)
        tc_rank_v = card_rank(tc_codes).long()
        tc_suit_v = card_suit(tc_codes).long()
        tc_empty_v = (tc_codes == 0)
        ts_exp4 = ts.unsqueeze(1).expand_as(tc_suit_v)
        oj_exp4 = off_jack_suit.unsqueeze(1).expand_as(tc_suit_v)
        is_trump_trick = (
            (~tc_empty_v) & (
                (tc_suit_v == ts_exp4) |
                (tc_rank_v == 11) |
                ((tc_rank_v == 12) & (tc_suit_v == oj_exp4))
            )
        )
        # effective rank for trick winner calc: trump cards keep rank, others get 0
        # Jack of trump gets +0.5 to beat off-jack (both rank 12)
        is_jack_of_trump_trick = (tc_rank_v == 12) & (tc_suit_v == ts_exp4)
        eff_rank = torch.where(is_trump_trick, tc_rank_v.float(),
                               torch.zeros_like(tc_rank_v.float()))
        eff_rank = eff_rank + 0.5 * is_jack_of_trump_trick.float()
        winner_rank = eff_rank.max(dim=1).values   # (N,) — 0 if trick empty
        trick_empty = tc_empty_v.all(dim=1)        # (N,)
        # Also boost jack of trump in hand for can_beat comparison
        is_jack_of_trump_hand = (h_rank_long == 12) & (h_suit_long == ts_exp)
        boosted_trump_ranks = trump_ranks + 0.5 * is_jack_of_trump_hand.float()
        can_beat = (boosted_trump_ranks.max(dim=1).values > winner_rank)
        feat4 = torch.where(trick_empty, torch.ones(self.N, device=self.device),
                            can_beat.float())

        # feat 5: partner_winning
        # Find who played the winning card
        tp = self.trick_players  # (N, 4) int8, -1 for empty
        trick_not_empty = ~tc_empty_v  # (N, 4)
        # effective rank per slot (0 for empty, with jack-of-trump tiebreaker)
        winner_slot = eff_rank.argmax(dim=1)  # (N,)
        winner_player = tp[game_idx, winner_slot]  # (N,) int8
        partner_winning = (winner_player.long() % 2 == cp % 2).float()
        # mask: 0 if trick is empty or not PLAYING
        feat5 = torch.where(trick_empty, torch.zeros(self.N, device=self.device),
                            partner_winning)

        # feat 6: am_high_bidder (always)
        feat6 = (self.current_high_bidder == self.current_player).float()

        # feat 7: bid_deficit (always) — max bid is 12 (double moon)
        my_team = (cp % 2).long()  # (N,) 0 or 1
        my_round_score = self.round_scores[game_idx, my_team].float()
        feat7 = (self.current_bid.float() - my_round_score).clamp(min=0) / 12.0

        # feat 8: tricks_remaining (always) — max hand size is 9 at deal, 6 after fill
        hand_occupied = (self.hands != 0)  # (N, 4, 10)
        hand_sizes = hand_occupied.sum(dim=2).float()  # (N, 4)
        feat8 = hand_sizes.max(dim=1).values / 9.0

        # feat 9: point_cards_remaining (always)
        total_scored = self.round_scores.float().sum(dim=1)  # (N,)
        feat9 = 1.0 - total_scored.clamp(max=9) / 9.0

        # Stack and apply phase guard (features 0-5 are 0 outside PLAYING)
        derived = torch.stack([
            feat0, feat1, feat2, feat3, feat4, feat5,  # phase-gated
            feat6, feat7, feat8, feat9,                 # always valid
        ], dim=1)  # (N, 10)
        derived[:, :6] *= is_playing.unsqueeze(1)

        # Concatenate in dict iteration order matching flatten_observation
        obs = torch.cat([
            hand_flat,           # 20
            pc_flat,             # 48
            scores_flat,         # 2
            round_scores_flat,   # 2
            trick_flat,          # 12
            scalars,             # 7
            pct_flat,            # 4
            mask_flat,           # 24
            derived,             # 10
        ], dim=1)  # (N, 129)
        return obs, mask

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _handle_bid(self, actions: torch.Tensor):
        """Handle bidding for all games in bidding phase."""
        m = self.phase == PHASE_BIDDING
        if not m.any():
            return

        # Actual bids (actions 11-18)
        is_bid = m & (actions >= 11) & (actions <= 18)
        self.current_bid = torch.where(
            is_bid, (actions - 6).to(torch.int8), self.current_bid
        )
        self.current_high_bidder = torch.where(
            is_bid, self.current_player, self.current_high_bidder
        )

        # Advance: if at dealer, bidding ends → choose suit phase
        at_dealer = m & (self.current_player == self.dealer)
        not_at_dealer = m & ~at_dealer

        self.phase = torch.where(
            at_dealer,
            torch.tensor(PHASE_CHOOSESUIT, dtype=torch.int8, device=self.device),
            self.phase,
        )
        self.current_player = torch.where(
            at_dealer, self.current_high_bidder, self.current_player
        )
        self.current_player = torch.where(
            not_at_dealer, (self.current_player + 1) % 4, self.current_player
        ).to(torch.int8)

    def _handle_choose_suit(self, actions: torch.Tensor):
        """Handle suit choice for games in choose-suit phase."""
        m = (self.phase == PHASE_CHOOSESUIT) & (
            self.current_player == self.current_high_bidder
        )
        if not m.any():
            return

        suit_chosen = m & (actions >= 19) & (actions <= 22)
        self.trump_suit = torch.where(
            suit_chosen, (actions - 19).to(torch.int8), self.trump_suit
        )
        self.phase = torch.where(
            suit_chosen,
            torch.tensor(PHASE_PLAYING, dtype=torch.int8, device=self.device),
            self.phase,
        )

        # Discard and fill for games that just chose suit
        if suit_chosen.any():
            self._discard_and_fill(suit_chosen)

    def _handle_play(self, actions: torch.Tensor):
        """Handle card play for games in playing phase."""
        m = (self.phase == PHASE_PLAYING) & (actions <= 9)
        if not m.any():
            return

        game_idx = torch.arange(self.N, device=self.device)[m]
        player_idx = self.current_player[m].long()
        slot_idx = actions[m].long()

        # Get the card from hand
        card = self.hands[game_idx, player_idx, slot_idx].clone()

        # Place in trick at playing_iterator position
        iter_pos = self.playing_iterator[m].long()
        self.trick_cards[game_idx, iter_pos] = card
        self.trick_players[game_idx, iter_pos] = self.current_player[m]

        # Remove from hand and compact (shift remaining cards left to fill gap)
        # Vectorized: gather-based shift avoids per-game Python loop + tensor clones
        M = len(game_idx)
        arange10 = torch.arange(10, device=self.device).unsqueeze(0).expand(M, -1)
        si_expanded = slot_idx.unsqueeze(1)  # (M, 1)
        # For positions >= removed slot, source is position + 1 (clamped to 9)
        gather_src = torch.where(
            arange10 >= si_expanded, (arange10 + 1).clamp(max=9), arange10
        ).long()
        hands_batch = self.hands[game_idx, player_idx]  # (M, 10)
        new_hands = hands_batch.gather(1, gather_src)
        new_hands[:, -1] = 0  # last slot always empty after removal
        self.hands[game_idx, player_idx] = new_hands

        # Track played cards
        pc_idx = self.played_cards_idx[m].long()
        self.played_cards[game_idx, pc_idx] = card
        self.played_cards_idx[m] += 1

    def _advance_after_play(self, actions: torch.Tensor):
        """After playing a card (or no valid play), advance iterator or resolve trick."""
        m = (self.phase == PHASE_PLAYING) & ((actions <= 9) | (actions == 23))
        if not m.any():
            return

        # Games where trick is complete (iterator was 3)
        trick_complete = m & (self.playing_iterator == 3)
        trick_continues = m & (self.playing_iterator < 3)

        # Advance for continuing tricks
        self.playing_iterator = torch.where(
            trick_continues,
            (self.playing_iterator + 1).to(torch.int8),
            self.playing_iterator,
        )
        self.current_player = torch.where(
            trick_continues,
            ((self.current_player + 1) % 4).to(torch.int8),
            self.current_player,
        )

        # Resolve completed tricks
        if trick_complete.any():
            self._resolve_tricks(trick_complete)
            # Trick winner leads next; if they have no valid plays, skip them
            winner_hands = self.hands[torch.arange(self.N, device=self.device)[trick_complete],
                                      self.current_player[trick_complete].long()]  # (M, 10)
            # Check if trick winner has valid plays (inline — _is_valid_play uses self.N shape)
            w_suits = card_suit(winner_hands)
            w_ranks = card_rank(winner_hands)
            w_trump = self.trump_suit[trick_complete].unsqueeze(1)
            w_oj = w_trump ^ 1
            w_valid = (winner_hands != 0) & (
                (w_suits == w_trump) | (winner_hands == JOKER_CODE) |
                ((w_ranks == 12) & (w_suits == w_oj))
            )
            no_valid = ~w_valid.any(dim=1)  # (M,) True if winner has no valid plays
            # For winners with no valid plays: advance player and set iterator=1
            tc_idx = torch.arange(self.N, device=self.device)[trick_complete]
            self.current_player[tc_idx[no_valid]] = ((self.current_player[tc_idx[no_valid]] + 1) % 4).to(torch.int8)
            self.playing_iterator[tc_idx[no_valid]] = 1

    def _handle_no_valid_play(self, actions: torch.Tensor):
        """Handle action 23 (no valid play) — check if round should end.

        Python env logic: if the trick is empty (playing_iterator==0) and
        no player in the game has any valid play, end the round.
        """
        m = (self.phase == PHASE_PLAYING) & (actions == 23)
        if not m.any():
            return

        # Only consider games where the trick is empty (no cards played yet)
        candidates = m & (self.playing_iterator == 0)
        if not candidates.any():
            return

        # Check all 4 players' hands across all N games — fully vectorized
        all_hands = self.hands  # (N, 4, 10)
        valid = self._is_valid_play(all_hands) & (all_hands != 0)  # (N, 4, 10)
        any_valid = valid.any(dim=2).any(dim=1)  # (N,)
        round_should_end = candidates & ~any_valid

        if round_should_end.any():
            self._end_round(round_should_end)

    # ------------------------------------------------------------------
    # Trick resolution
    # ------------------------------------------------------------------

    def _resolve_tricks(self, mask: torch.Tensor):
        """Resolve completed tricks for masked games."""
        if not mask.any():
            return

        g = torch.arange(self.N, device=self.device)[mask]
        cards = self.trick_cards[mask]       # (M, 4)
        players = self.trick_players[mask]   # (M, 4)

        suits = card_suit(cards)  # (M, 4)
        ranks = card_rank(cards)  # (M, 4)

        trump = self.trump_suit[mask].unsqueeze(1)  # (M, 1)

        # Determine which cards are trump
        is_trump = (suits == trump)
        is_joker = (cards == JOKER_CODE)
        off_jack_suit = trump ^ 1
        is_off_jack = (ranks == 12) & (suits == off_jack_suit)
        is_trump = is_trump | is_joker | is_off_jack

        # Winner: highest rank among trump cards
        # Non-trump cards get effective rank 0
        # Jack of trump (rank 12, suit == trump) gets +0.5 to beat off-jack (rank 12)
        is_jack_of_trump = (ranks == 12) & (suits == trump)
        eff_rank = torch.where(is_trump, ranks.float(), torch.zeros(len(g), 4, device=self.device))
        eff_rank = eff_rank + 0.5 * is_jack_of_trump.float()
        winner_pos = eff_rank.argmax(dim=1)  # (M,)
        winner_player = players.gather(
            1, winner_pos.unsqueeze(1).long()
        ).squeeze(1)  # (M,)

        # Points
        points = self._card_points_lookup(cards)  # (M, 4)
        winning_team = (winner_player % 2).long()  # (M,)

        # 2-of-trump: goes to the team that played it, not trick winner
        is_two = (ranks == 2) & (cards != 0)
        two_points = points * is_two.to(torch.int8)
        other_points = points * (~is_two).to(torch.int8)

        # Scatter 2-of-trump points to playing team
        playing_team = (players % 2).long()  # (M, 4)
        for t in range(2):
            team_two = (two_points * (playing_team == t).to(torch.int8)).sum(dim=1)
            self.round_scores[g, t] += team_two.to(torch.int8)
            self.last_trick_points[g, t] = team_two.to(torch.int8)

        # Other points go to trick winner's team
        total_other = other_points.sum(dim=1)  # (M,)
        for t in range(2):
            wins_t = winning_team == t
            self.round_scores[g[wins_t], t] += total_other[wins_t].to(torch.int8)
            self.last_trick_points[g[wins_t], t] += total_other[wins_t].to(torch.int8)

        # Store winner, reset trick, set next leader
        self.trick_winner[mask] = winner_player.to(torch.int8)
        self.trick_cards[mask] = 0
        self.trick_players[mask] = -1
        self.playing_iterator[mask] = 0
        self.current_player[mask] = winner_player.to(torch.int8)

    # ------------------------------------------------------------------
    # Round end / scoring
    # ------------------------------------------------------------------

    def _end_round(self, mask: torch.Tensor):
        """End the round: evaluate bids, update scores, start new round."""
        if not mask.any():
            return

        g = torch.arange(self.N, device=self.device)[mask]
        bid = self.current_bid[mask].long()        # (M,)
        bidder = self.current_high_bidder[mask]     # (M,)
        bidder_team = (bidder % 2).long()           # (M,)
        other_team = 1 - bidder_team                # (M,)
        rs = self.round_scores[mask]                # (M, 2)

        # Bidder team's round score
        bidder_rs = rs.gather(1, bidder_team.unsqueeze(1)).squeeze(1)  # (M,)
        other_rs = rs.gather(1, other_team.unsqueeze(1)).squeeze(1)    # (M,)

        # Normal bids (current_bid <= 10): make bid → +round_score, miss → -bid
        normal = bid <= 10
        made_bid = normal & (bidder_rs >= bid)
        missed_bid = normal & (bidder_rs < bid)

        # Moon bids (11=shoot moon, 12=double shoot moon)
        is_moon = bid == 11
        is_double_moon = bid == 12
        moon_made = (is_moon | is_double_moon) & (bidder_rs == 10)
        moon_missed = (is_moon | is_double_moon) & (bidder_rs != 10)

        # Compute score deltas for bidder team
        bidder_delta = torch.zeros(mask.sum().item(), dtype=torch.int16, device=self.device)
        bidder_delta = torch.where(made_bid, bidder_rs.to(torch.int16), bidder_delta)
        bidder_delta = torch.where(missed_bid, -bid.to(torch.int16), bidder_delta)
        bidder_delta = torch.where(
            moon_made & is_moon,
            torch.tensor(20, dtype=torch.int16, device=self.device),
            bidder_delta,
        )
        bidder_delta = torch.where(
            moon_made & is_double_moon,
            torch.tensor(40, dtype=torch.int16, device=self.device),
            bidder_delta,
        )
        bidder_delta = torch.where(
            moon_missed & is_moon,
            torch.tensor(-20, dtype=torch.int16, device=self.device),
            bidder_delta,
        )
        bidder_delta = torch.where(
            moon_missed & is_double_moon,
            torch.tensor(-40, dtype=torch.int16, device=self.device),
            bidder_delta,
        )

        # Apply bidder team score
        for t in range(2):
            is_t = bidder_team == t
            self.scores[g[is_t], t] += bidder_delta[is_t]

        # Other team always gets their round score
        for t in range(2):
            is_t = other_team == t
            self.scores[g[is_t], t] += other_rs[is_t].to(torch.int16)

        # Save round_scores before reset clears them (used by _calculate_rewards)
        self._saved_round_scores[mask] = self.round_scores[mask]

        # Check for game end before resetting round
        self._check_game_end(mask)

        # Reset round for games that aren't done
        still_playing = mask.clone()
        still_playing[mask] = ~self.done[mask]
        if still_playing.any():
            self._reset_round(still_playing)

    def _check_game_end(self, mask: torch.Tensor):
        """Check win conditions for masked games."""
        if not mask.any():
            return

        g = torch.arange(self.N, device=self.device)[mask]
        s = self.scores[mask]  # (M, 2)
        t = self.win_threshold

        bidder_team = (self.current_high_bidder[mask] % 2).long()

        # Too many rounds
        too_many = self.num_rounds_played[mask] >= 50

        # Score difference >= threshold
        score_diff = (s[:, 0] - s[:, 1]).abs() >= t

        # Team reached threshold AND they are the bidding team
        team0_wins = (s[:, 0] >= t) & (bidder_team == 0)
        team1_wins = (s[:, 1] >= t) & (bidder_team == 1)

        game_over = too_many | score_diff | team0_wins | team1_wins
        self.done[g[game_over]] = True

    # ------------------------------------------------------------------
    # Discard and fill (CPU fallback)
    # ------------------------------------------------------------------

    def _discard_and_fill(self, mask: torch.Tensor):
        """Discard non-trump cards and fill hands from deck.

        Falls back to CPU for the complex variable-length logic.
        Only called once per round when trump is chosen.
        """
        if not mask.any():
            return

        indices = torch.where(mask)[0]

        # Pull state to CPU
        hands_cpu = self.hands[mask].cpu().numpy().copy()    # (M, 4, 10)
        deck_cpu = self.deck[mask].cpu().numpy().copy()      # (M, 54)
        trump_cpu = self.trump_suit[mask].cpu().numpy().copy()      # (M,)
        dealer_cpu = self.dealer[mask].cpu().numpy().copy()          # (M,)
        bidder_cpu = self.current_high_bidder[mask].cpu().numpy().copy()  # (M,)
        cards_taken_cpu = self.player_cards_taken[mask].cpu().numpy().copy()  # (M, 4)

        M = len(indices)
        for i in range(M):
            self._discard_and_fill_single(
                hands_cpu[i], deck_cpu[i], trump_cpu[i],
                dealer_cpu[i], bidder_cpu[i], cards_taken_cpu[i],
            )

        # Push back
        self.hands[mask] = torch.from_numpy(hands_cpu).to(self.device)
        self.deck[mask] = torch.from_numpy(deck_cpu).to(self.device)
        self.player_cards_taken[mask] = torch.from_numpy(cards_taken_cpu).to(self.device)

    def _discard_and_fill_single(
        self, hands: np.ndarray, deck: np.ndarray, trump: int,
        dealer: int, bidder: int, cards_taken: np.ndarray,
    ):
        """CPU logic for a single game's discard-and-fill.

        hands: (4, 10) int8 array of card codes
        deck: (54,) int8 array
        trump: int (0-3)
        dealer: int (0-3)
        bidder: int (0-3)
        cards_taken: (4,) int8 array
        """
        partner = (bidder + 2) % 4

        def is_valid(card_code):
            if card_code == 0:
                return False
            suit = int(card_code) // 16
            rank = int(card_code) % 16
            if suit == trump:
                return True
            if card_code == JOKER_CODE:
                return True
            # Off-jack
            if rank == 12 and suit == (trump ^ 1):
                return True
            return False

        def compact_hand(player):
            """Move non-zero cards to the front of the hand."""
            h = hands[player]
            nonzero = h[h != 0]
            hands[player] = 0
            hands[player, :len(nonzero)] = nonzero

        def find_deck_top():
            """Find first non-zero card in deck (from the end, matching pop behavior)."""
            for j in range(len(deck) - 1, -1, -1):
                if deck[j] != 0:
                    return j
            return -1

        def deck_pop():
            """Pop the top card from the deck (from the end)."""
            j = find_deck_top()
            if j < 0:
                return 0
            card = deck[j]
            deck[j] = 0
            return card

        # Phase 1: Discard non-playable cards (in dealer order)
        for pi in range(4):
            p = (pi + dealer) % 4
            for slot in range(10):
                if hands[p, slot] != 0 and not is_valid(hands[p, slot]):
                    hands[p, slot] = 0
            compact_hand(p)

        # Phase 2: Fill to 6 cards from deck (in dealer order)
        for pi in range(4):
            p = (pi + dealer) % 4
            num_cards = int(np.sum(hands[p] != 0))
            while num_cards < 6:
                card = deck_pop()
                if card == 0:
                    break  # deck empty
                # Find first empty slot
                for slot in range(10):
                    if hands[p, slot] == 0:
                        hands[p, slot] = card
                        cards_taken[p] += 1
                        break
                num_cards += 1

        # Phase 3: Bidder and partner swap invalid drawn cards
        # Python does list.remove(bad) + list.append(replacement),
        # so the replacement goes to the end. We must match this ordering.
        # Re-scan after each swap since compact_hand changes slot indices.
        for p in [bidder, partner]:
            while True:
                # Find first invalid card in hand
                bad_slot = -1
                for s in range(10):
                    if hands[p, s] != 0 and not is_valid(hands[p, s]):
                        bad_slot = s
                        break
                if bad_slot < 0:
                    break
                # Find a valid replacement in deck (forward search, matching Python)
                replacement_idx = -1
                for j in range(len(deck)):
                    if deck[j] != 0 and is_valid(deck[j]):
                        replacement_idx = j
                        break
                if replacement_idx < 0:
                    break  # no valid replacements left
                replacement = deck[replacement_idx]
                deck[replacement_idx] = 0
                # Remove the bad card (shift left) and append replacement
                hands[p, bad_slot] = 0
                compact_hand(p)
                # Append at first empty slot (at the end after compact)
                for s in range(10):
                    if hands[p, s] == 0:
                        hands[p, s] = replacement
                        break

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _calculate_rewards(self) -> torch.Tensor:
        """Calculate rewards for both teams.

        Returns: (N, 2) float tensor — rewards[:, t] is team t's reward.

        Reward components:
        - Per-trick: points won this step (dense signal during play)
        - Round-end adjustment: score_delta - round_scores (non-zero only
          when bid outcome differs from trick points, e.g. getting set)
        - Game-end bonus: +/-100 for winning/losing
        """
        rewards = torch.zeros(self.N, 2, device=self.device)

        # Trick-level: points each team won minus opponent
        trick_pts = self.last_trick_points.float()  # (N, 2)
        rewards[:, 0] = trick_pts[:, 0] - trick_pts[:, 1]
        rewards[:, 1] = trick_pts[:, 1] - trick_pts[:, 0]

        # Round-end bid adjustment: actual score change minus trick points
        # already given during the round. Non-zero only when bidder gets set
        # or makes a moon bid.
        score_delta = (self.scores - self.scores_before).float()  # (N, 2)
        saved_rs = self._saved_round_scores.float()  # (N, 2)
        adjustment = score_delta - saved_rs  # (N, 2)
        rewards[:, 0] += adjustment[:, 0] - adjustment[:, 1]
        rewards[:, 1] += adjustment[:, 1] - adjustment[:, 0]

        # Game-end bonus for both teams
        if self.done.any():
            done_mask = self.done
            s = self.scores[done_mask]
            bidder_team = (self.current_high_bidder[done_mask] % 2).long()

            # Team 0 wins if higher score, or reached threshold as bidder
            team0_wins = (s[:, 0] > s[:, 1]) | (
                (s[:, 0] >= self.win_threshold) & (bidder_team == 0)
            )
            bonus = torch.where(
                team0_wins,
                torch.tensor(100.0, device=self.device),
                torch.tensor(-100.0, device=self.device),
            )
            rewards[done_mask, 0] += bonus
            rewards[done_mask, 1] -= bonus

        return rewards

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, actions: torch.Tensor):
        """Step all N games with the given actions.

        actions: (N,) int tensor of action indices
        Returns: (observations, rewards, dones) where rewards is (N, 2) for both teams
        """
        actions = actions.to(torch.int8)

        # Mask out done games — set their action to a no-op that won't match any handler
        actions = torch.where(self.done, torch.tensor(-1, dtype=torch.int8, device=self.device), actions)

        # Save scores for reward calculation
        self.scores_before = self.scores.clone()
        self.last_trick_points[:] = 0
        self._saved_round_scores[:] = 0

        # Execute actions by phase
        self._handle_bid(actions)
        self._handle_choose_suit(actions)
        self._handle_play(actions)
        self._handle_no_valid_play(actions)
        self._advance_after_play(actions)

        # Build outputs
        rewards = self._calculate_rewards()
        obs, _ = self.get_observations()

        return obs, rewards, self.done.clone()
