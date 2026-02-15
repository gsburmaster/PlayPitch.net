"""
Parity tests: VectorizedPitchEnv vs PitchEnv.

V1: Deterministic scenario parity (specific game states)
V2: Random game parity (play full games with same random actions)
V5: Edge cases
V6: Observation layout match
"""

import unittest

import numpy as np
import torch

from pitch_env import PitchEnv, Suit, Phase, Card
from train import flatten_observation
from vectorized_env import (
    VectorizedPitchEnv,
    JOKER_CODE,
    TEMPLATE_DECK,
    card_suit,
    card_rank,
    encode_card,
    PHASE_BIDDING,
    PHASE_CHOOSESUIT,
    PHASE_PLAYING,
)


def encode_py_card(card: Card) -> int:
    """Encode a Python Card as a vectorized env int8 code."""
    if card.suit is None:
        return JOKER_CODE
    return card.suit.value * 16 + card.rank


def inject_deck(py_env: PitchEnv, vec_env: VectorizedPitchEnv, game_idx: int = 0):
    """Copy PitchEnv's deck into VectorizedPitchEnv (post-deal remaining deck).

    Call AFTER both envs have been reset and dealt, to sync the remaining deck.
    This doesn't work for full parity — use inject_full_state instead.
    """
    pass  # Not used; see inject_full_state


def setup_synced_envs(seed: int = 42):
    """Create a PitchEnv and a VectorizedPitchEnv(N=1) with identical state.

    We reset the Python env, then manually copy its full state into the vec env.
    """
    py_env = PitchEnv()
    py_env.reset(seed=seed)

    vec_env = VectorizedPitchEnv(1, torch.device("cpu"))
    # Don't call reset_all — we'll inject state manually
    vec_env.done[0] = False

    # Copy deck
    vec_deck = torch.zeros(54, dtype=torch.int8)
    for i, card in enumerate(py_env.deck):
        vec_deck[i] = encode_py_card(card)
    vec_env.deck[0] = vec_deck

    # Copy hands
    for p in range(4):
        for s, card in enumerate(py_env.hands[p]):
            vec_env.hands[0, p, s] = encode_py_card(card)

    # Copy scalars
    vec_env.phase[0] = py_env.phase.value
    vec_env.dealer[0] = py_env.dealer
    vec_env.current_player[0] = py_env.current_player
    vec_env.current_bid[0] = py_env.current_bid
    vec_env.current_high_bidder[0] = py_env.current_high_bidder
    vec_env.trump_suit[0] = py_env.trump_suit.value if py_env.trump_suit is not None else -1
    vec_env.playing_iterator[0] = py_env.playing_iterator
    vec_env.scores[0, 0] = py_env.scores[0]
    vec_env.scores[0, 1] = py_env.scores[1]
    vec_env.round_scores[0, 0] = py_env.round_scores[0]
    vec_env.round_scores[0, 1] = py_env.round_scores[1]
    vec_env.num_rounds_played[0] = py_env.number_of_rounds_played
    for p in range(4):
        vec_env.player_cards_taken[0, p] = py_env.player_cards_taken[p]

    return py_env, vec_env


class TestCardEncoding(unittest.TestCase):
    """Basic tests for card encoding/decoding."""

    def test_encode_decode_roundtrip(self):
        for suit in range(4):
            for rank in range(2, 16):
                code = encode_card(suit, rank)
                self.assertEqual(card_suit(torch.tensor(code)).item(), suit)
                self.assertEqual(card_rank(torch.tensor(code)).item(), rank)

    def test_joker_encoding(self):
        self.assertEqual(JOKER_CODE, 75)
        self.assertEqual(card_suit(torch.tensor(JOKER_CODE)).item(), 4)
        self.assertEqual(card_rank(torch.tensor(JOKER_CODE)).item(), 11)

    def test_template_deck_size(self):
        self.assertEqual(len(TEMPLATE_DECK), 54)

    def test_empty_card_is_zero(self):
        self.assertEqual(card_suit(torch.tensor(0)).item(), 0)
        self.assertEqual(card_rank(torch.tensor(0)).item(), 0)


class TestResetAndDeal(unittest.TestCase):
    """Test initialization, deck creation, and dealing."""

    def test_reset_all_shapes(self):
        env = VectorizedPitchEnv(8, torch.device("cpu"))
        obs = env.reset_all()
        self.assertEqual(obs.shape, (8, 119))
        self.assertTrue((env.phase == PHASE_BIDDING).all())
        self.assertFalse(env.done.any())

    def test_nine_cards_per_player(self):
        env = VectorizedPitchEnv(4, torch.device("cpu"))
        env.reset_all()
        cards_per_player = (env.hands != 0).sum(dim=2)  # (4, 4)
        self.assertTrue((cards_per_player == 9).all(),
                        f"Expected 9 cards each, got:\n{cards_per_player}")

    def test_no_duplicate_cards_in_game(self):
        """Each game should have 54 unique cards across hands + deck."""
        env = VectorizedPitchEnv(4, torch.device("cpu"))
        env.reset_all()
        for g in range(4):
            all_cards = []
            for p in range(4):
                for s in range(10):
                    c = env.hands[g, p, s].item()
                    if c != 0:
                        all_cards.append(c)
            for d in range(54):
                c = env.deck[g, d].item()
                if c != 0:
                    all_cards.append(c)
            # Should have exactly 54 cards (36 in hands + 18 in deck)
            self.assertEqual(len(all_cards), 54, f"Game {g}: expected 54 cards, got {len(all_cards)}")

    def test_dealer_is_valid(self):
        env = VectorizedPitchEnv(100, torch.device("cpu"))
        env.reset_all()
        self.assertTrue((env.dealer >= 0).all() and (env.dealer <= 3).all())
        self.assertTrue(((env.current_player) == ((env.dealer + 1) % 4)).all())


class TestValidityCheck(unittest.TestCase):
    """Test _is_valid_play against known scenarios."""

    def test_trump_is_valid(self):
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        env.trump_suit[0] = 0  # Hearts
        card = torch.tensor([[[encode_card(0, 5)]]], dtype=torch.int8)  # Hearts 5
        # Need to reshape for _is_valid_play which expects first dim = N
        env_card = torch.tensor([[encode_card(0, 5)]], dtype=torch.int8)
        result = env._is_valid_play(env_card)
        self.assertTrue(result[0, 0].item())

    def test_non_trump_is_invalid(self):
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        env.trump_suit[0] = 0  # Hearts
        env_card = torch.tensor([[encode_card(2, 5)]], dtype=torch.int8)  # Clubs 5
        result = env._is_valid_play(env_card)
        self.assertFalse(result[0, 0].item())

    def test_joker_is_valid(self):
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        env.trump_suit[0] = 2  # Clubs
        env_card = torch.tensor([[JOKER_CODE]], dtype=torch.int8)
        result = env._is_valid_play(env_card)
        self.assertTrue(result[0, 0].item())

    def test_off_jack_is_valid(self):
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        env.trump_suit[0] = 0  # Hearts
        # Off-jack = Diamonds Jack (suit 1, rank 12)
        env_card = torch.tensor([[encode_card(1, 12)]], dtype=torch.int8)
        result = env._is_valid_play(env_card)
        self.assertTrue(result[0, 0].item())

    def test_off_jack_wrong_color(self):
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        env.trump_suit[0] = 0  # Hearts
        # Clubs Jack (suit 2, rank 12) is NOT the off-jack for Hearts
        env_card = torch.tensor([[encode_card(2, 12)]], dtype=torch.int8)
        result = env._is_valid_play(env_card)
        self.assertFalse(result[0, 0].item())

    def test_empty_card_is_invalid(self):
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        env.trump_suit[0] = 0
        env_card = torch.tensor([[0]], dtype=torch.int8)
        result = env._is_valid_play(env_card)
        self.assertFalse(result[0, 0].item())


class TestActionMask(unittest.TestCase):
    """Test action mask generation."""

    def test_bidding_mask_initial(self):
        """First bidder (not dealer) should be able to pass and bid 5-moon."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        mask = env._get_action_mask()[0]
        # Should be in bidding, not dealer
        self.assertEqual(env.phase[0].item(), PHASE_BIDDING)
        self.assertNotEqual(env.current_player[0].item(), env.dealer[0].item())
        # Can pass (10) and bid 5-moon (11-17)
        self.assertEqual(mask[10].item(), 1, "Should be able to pass")
        for i in range(11, 18):
            self.assertEqual(mask[i].item(), 1, f"Should be able to bid action {i}")
        # Can't double moon (only dealer when someone bid moon)
        self.assertEqual(mask[18].item(), 0)

    def test_playing_mask_valid_cards(self):
        """In playing phase, mask should match hand validity."""
        py_env, vec_env = setup_synced_envs(seed=42)

        # Push through bidding and suit choice manually
        # Player bids, everyone else passes, then choose suit
        dealer = py_env.dealer
        bidder = (dealer + 1) % 4

        # Bidder bids 5 (action 11), others pass (action 10)
        for _ in range(4):
            py_obs = py_env._get_observation()
            if py_env.current_player == bidder:
                py_env.step(11, py_obs)  # bid 5
            else:
                py_env.step(10, py_obs)  # pass

        # Choose suit (action 19 = hearts)
        py_obs = py_env._get_observation()
        py_env.step(19, py_obs)

        # Now sync vec_env to match
        vec_env.phase[0] = PHASE_PLAYING
        vec_env.trump_suit[0] = 0  # Hearts
        vec_env.current_bid[0] = py_env.current_bid
        vec_env.current_high_bidder[0] = py_env.current_high_bidder
        vec_env.current_player[0] = py_env.current_player

        # Copy post-discard hands
        vec_env.hands[0] = 0
        for p in range(4):
            for s, card in enumerate(py_env.hands[p]):
                vec_env.hands[0, p, s] = encode_py_card(card)

        # Compare masks
        py_mask = py_env._get_action_mask()
        vec_mask = vec_env._get_action_mask()[0].numpy()

        np.testing.assert_array_equal(
            py_mask, vec_mask,
            err_msg="Playing phase masks don't match"
        )


class TestObservationLayout(unittest.TestCase):
    """V6: Verify observation tensor matches flatten_observation() exactly."""

    def test_initial_obs_layout(self):
        """After reset, obs from vec env should match Python env layout."""
        py_env, vec_env = setup_synced_envs(seed=42)

        py_obs = py_env._get_observation()
        py_flat = flatten_observation(py_obs)

        vec_flat = vec_env.get_observations()[0].numpy()

        self.assertEqual(len(py_flat), len(vec_flat),
                         f"Length mismatch: py={len(py_flat)} vec={len(vec_flat)}")

        for i in range(len(py_flat)):
            self.assertAlmostEqual(
                py_flat[i], vec_flat[i], places=4,
                msg=f"Mismatch at index {i}: py={py_flat[i]} vec={vec_flat[i]}"
            )


class TestBiddingParity(unittest.TestCase):
    """V1: Deterministic bidding scenario parity."""

    def test_pass_advances_player(self):
        """Passing should advance current_player by 1."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        cp_before = env.current_player[0].item()
        actions = torch.tensor([10], dtype=torch.int8)  # pass
        env._handle_bid(actions)
        cp_after = env.current_player[0].item()
        expected = (cp_before + 1) % 4
        self.assertEqual(cp_after, expected)

    def test_bid_updates_state(self):
        """A bid should update current_bid and current_high_bidder."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        cp = env.current_player[0].item()
        actions = torch.tensor([13], dtype=torch.int8)  # bid 7
        env._handle_bid(actions)
        self.assertEqual(env.current_bid[0].item(), 7)  # 13 - 6 = 7
        self.assertEqual(env.current_high_bidder[0].item(), cp)

    def test_dealer_ends_bidding(self):
        """When dealer bids/passes, phase should move to CHOOSESUIT."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        # Advance to dealer
        for _ in range(3):
            env._handle_bid(torch.tensor([10], dtype=torch.int8))  # pass

        # Now current_player should be dealer
        self.assertEqual(env.current_player[0].item(), env.dealer[0].item())

        # Dealer bids
        env._handle_bid(torch.tensor([11], dtype=torch.int8))  # bid 5
        self.assertEqual(env.phase[0].item(), PHASE_CHOOSESUIT)


class TestRandomGameParity(unittest.TestCase):
    """V2: Play complete games with random actions on both envs, compare outcomes."""

    def _play_one_game(self, seed: int):
        """Play one full game on both envs with the same random actions.

        Returns (py_scores, vec_scores, num_steps, diverged_at).
        """
        py_env, vec_env = setup_synced_envs(seed=seed)

        rng = np.random.RandomState(seed + 10000)
        py_obs = py_env._get_observation()
        max_steps = 5000

        for step in range(max_steps):
            # Get masks from both
            py_mask = py_obs["action_mask"]
            vec_mask = vec_env._get_action_mask()[0].numpy()

            # Compare masks
            if not np.array_equal(py_mask, vec_mask):
                return None, None, step, f"Mask mismatch at step {step}"

            # Pick random valid action
            valid = np.where(py_mask == 1)[0]
            if len(valid) == 0:
                return None, None, step, f"No valid actions at step {step}"
            action = int(rng.choice(valid))

            # Step both
            py_obs, py_r, py_done, _, _ = py_env.step(action, py_obs)
            vec_obs, vec_r, vec_done = vec_env.step(
                torch.tensor([action], dtype=torch.int64)
            )

            if py_done:
                return py_env.scores, vec_env.scores[0].tolist(), step + 1, None

            # Sync state that may have drifted due to round resets
            # (the vec env shuffles its own deck on round reset)
            if vec_env.phase[0].item() == PHASE_BIDDING and py_env.phase == Phase.BIDDING:
                # A new round started — resync hands and deck
                if py_env.number_of_rounds_played != vec_env.num_rounds_played[0].item():
                    # Rounds are out of sync
                    return None, None, step, "Round count mismatch"

                # If both just started a new round, we need to resync the random state
                # because they shuffled independently. Copy Python state into vec.
                vec_env.hands[0] = 0
                for p in range(4):
                    for s, card in enumerate(py_env.hands[p]):
                        vec_env.hands[0, p, s] = encode_py_card(card)
                vec_env.deck[0] = 0
                for i, card in enumerate(py_env.deck):
                    vec_env.deck[0, i] = encode_py_card(card)
                vec_env.dealer[0] = py_env.dealer
                vec_env.current_player[0] = py_env.current_player
                vec_env.current_bid[0] = py_env.current_bid
                vec_env.current_high_bidder[0] = py_env.current_high_bidder
                vec_env.round_scores[0] = 0
                vec_env.played_cards[0] = 0
                vec_env.played_cards_idx[0] = 0
                vec_env.playing_iterator[0] = 0
                vec_env.trick_cards[0] = 0
                vec_env.trick_players[0] = -1
                for p in range(4):
                    vec_env.player_cards_taken[0, p] = py_env.player_cards_taken[p]

        return None, None, max_steps, "Game didn't finish"

    def test_random_games_parity(self):
        """Play 100 random games and verify scores match."""
        num_games = 100
        failures = []

        for seed in range(num_games):
            py_scores, vec_scores, steps, err = self._play_one_game(seed)
            if err is not None:
                failures.append(f"Game {seed}: {err}")
                continue
            if py_scores != vec_scores:
                failures.append(
                    f"Game {seed}: score mismatch py={py_scores} vec={vec_scores}"
                )

        if failures:
            self.fail(
                f"{len(failures)}/{num_games} games failed:\n"
                + "\n".join(failures[:10])
            )


class TestTrickResolution(unittest.TestCase):
    """Test trick resolution scenarios."""

    def _setup_playing_env(self, trump_suit=0):
        """Create a vec env in playing phase with given trump."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.done[0] = False
        env.phase[0] = PHASE_PLAYING
        env.trump_suit[0] = trump_suit
        env.current_player[0] = 0
        env.playing_iterator[0] = 0
        env.scores[0] = 0
        env.round_scores[0] = 0
        return env

    def test_highest_trump_wins(self):
        """Ace of trump should beat lower trump cards."""
        env = self._setup_playing_env(trump_suit=0)  # Hearts
        # Set up 4 players with one card each
        env.hands[0, 0, 0] = encode_card(0, 10)  # Hearts 10
        env.hands[0, 1, 0] = encode_card(0, 5)   # Hearts 5
        env.hands[0, 2, 0] = encode_card(0, 15)  # Hearts Ace
        env.hands[0, 3, 0] = encode_card(0, 3)   # Hearts 3

        # Play all 4 cards
        for _ in range(4):
            actions = torch.tensor([0], dtype=torch.int64)
            env.step(actions)

        # Ace (player 2) should have won
        self.assertEqual(env.trick_winner[0].item(), 2)

    def test_two_of_trump_scores_to_player(self):
        """2-of-trump point goes to the team that played it."""
        env = self._setup_playing_env(trump_suit=0)  # Hearts
        # Player 0 (team 0) plays 2, Player 1 (team 1) plays Ace
        env.hands[0, 0, 0] = encode_card(0, 2)   # Hearts 2
        env.hands[0, 1, 0] = encode_card(0, 15)  # Hearts Ace
        env.hands[0, 2, 0] = encode_card(0, 5)   # Hearts 5
        env.hands[0, 3, 0] = encode_card(0, 6)   # Hearts 6

        for _ in range(4):
            env.step(torch.tensor([0], dtype=torch.int64))

        # Ace wins → player 1 (team 1)
        self.assertEqual(env.trick_winner[0].item(), 1)
        # 2's point → team 0 (player 0's team)
        self.assertEqual(env.round_scores[0, 0].item(), 1)
        # Ace's point → team 1 (trick winner)
        self.assertEqual(env.round_scores[0, 1].item(), 1)


class TestEdgeCases(unittest.TestCase):
    """V5: Known tricky scenarios."""

    def test_off_jack_wins_trick(self):
        """Off-jack should be treated as trump and rank correctly."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.done[0] = False
        env.phase[0] = PHASE_PLAYING
        env.trump_suit[0] = 0  # Hearts
        env.current_player[0] = 0
        env.playing_iterator[0] = 0

        # Off-jack for Hearts = Diamonds Jack (suit 1, rank 12)
        env.hands[0, 0, 0] = encode_card(1, 12)  # Off-jack (rank 12)
        env.hands[0, 1, 0] = encode_card(0, 5)   # Hearts 5
        env.hands[0, 2, 0] = encode_card(0, 6)   # Hearts 6
        env.hands[0, 3, 0] = encode_card(0, 7)   # Hearts 7

        for _ in range(4):
            env.step(torch.tensor([0], dtype=torch.int64))

        # Off-jack (rank 12) beats 5, 6, 7
        self.assertEqual(env.trick_winner[0].item(), 0)

    def test_joker_scores_point(self):
        """Joker should score 1 point for the trick winner."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.done[0] = False
        env.phase[0] = PHASE_PLAYING
        env.trump_suit[0] = 0  # Hearts
        env.current_player[0] = 0
        env.playing_iterator[0] = 0

        env.hands[0, 0, 0] = JOKER_CODE             # Joker (rank 11)
        env.hands[0, 1, 0] = encode_card(0, 15)     # Hearts Ace (rank 15)
        env.hands[0, 2, 0] = encode_card(0, 5)      # Hearts 5
        env.hands[0, 3, 0] = encode_card(0, 6)      # Hearts 6

        for _ in range(4):
            env.step(torch.tensor([0], dtype=torch.int64))

        # Ace (rank 15) wins
        self.assertEqual(env.trick_winner[0].item(), 1)
        # Joker (1pt) + Ace (1pt) = 2 points to team 1
        self.assertEqual(env.round_scores[0, 1].item(), 2)


if __name__ == "__main__":
    unittest.main()
