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

        vec_obs, _ = vec_env.get_observations()
        vec_flat = vec_obs[0].numpy()

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


class TestEndRound(unittest.TestCase):
    """Test _end_round() bid evaluation logic."""

    def _setup_round_end(self, bid, bidder, round_scores, trump_suit=0):
        """Create an env ready for _end_round with specific bid/scores."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.done[0] = False
        env.phase[0] = PHASE_PLAYING
        env.trump_suit[0] = trump_suit
        env.current_bid[0] = bid
        env.current_high_bidder[0] = bidder
        env.round_scores[0, 0] = round_scores[0]
        env.round_scores[0, 1] = round_scores[1]
        env.scores[0] = 0
        env.num_rounds_played[0] = 0
        # Need a full deck for _reset_round after _end_round
        env._create_and_shuffle_decks(torch.ones(1, dtype=torch.bool))
        return env

    def test_normal_bid_made(self):
        """Bidder team scores round_score when bid is made."""
        env = self._setup_round_end(bid=7, bidder=0, round_scores=[8, 2])
        mask = torch.ones(1, dtype=torch.bool)
        env._end_round(mask)
        # Bidder (player 0, team 0) made bid (8 >= 7) → +8
        # Other team (team 1) → +2
        self.assertEqual(env.scores[0, 0].item(), 8)
        self.assertEqual(env.scores[0, 1].item(), 2)

    def test_normal_bid_missed(self):
        """Bidder team gets -bid when bid is missed (set)."""
        env = self._setup_round_end(bid=7, bidder=0, round_scores=[5, 5])
        mask = torch.ones(1, dtype=torch.bool)
        env._end_round(mask)
        # Bidder (team 0) missed (5 < 7) → -7
        # Other team → +5
        self.assertEqual(env.scores[0, 0].item(), -7)
        self.assertEqual(env.scores[0, 1].item(), 5)

    def test_moon_made(self):
        """Shoot the moon made → +20 for bidder team."""
        env = self._setup_round_end(bid=11, bidder=1, round_scores=[0, 10])
        mask = torch.ones(1, dtype=torch.bool)
        env._end_round(mask)
        # Bidder (player 1, team 1) made moon (10==10) → +20
        # Other team (team 0) → +0
        self.assertEqual(env.scores[0, 1].item(), 20)
        self.assertEqual(env.scores[0, 0].item(), 0)

    def test_moon_missed(self):
        """Shoot the moon missed → -20 for bidder team."""
        env = self._setup_round_end(bid=11, bidder=0, round_scores=[8, 2])
        mask = torch.ones(1, dtype=torch.bool)
        env._end_round(mask)
        # Bidder (team 0) missed moon (8!=10) → -20
        # Other team → +2
        self.assertEqual(env.scores[0, 0].item(), -20)
        self.assertEqual(env.scores[0, 1].item(), 2)

    def test_double_moon_made(self):
        """Double shoot the moon made → +40."""
        env = self._setup_round_end(bid=12, bidder=2, round_scores=[10, 0])
        mask = torch.ones(1, dtype=torch.bool)
        env._end_round(mask)
        # Bidder (player 2, team 0) → +40
        self.assertEqual(env.scores[0, 0].item(), 40)
        self.assertEqual(env.scores[0, 1].item(), 0)

    def test_double_moon_missed(self):
        """Double shoot the moon missed → -40."""
        env = self._setup_round_end(bid=12, bidder=3, round_scores=[2, 8])
        mask = torch.ones(1, dtype=torch.bool)
        env._end_round(mask)
        # Bidder (player 3, team 1) missed → -40
        # Other team (team 0) → +2
        self.assertEqual(env.scores[0, 1].item(), -40)
        self.assertEqual(env.scores[0, 0].item(), 2)

    def test_other_team_always_gets_round_score(self):
        """Non-bidding team always gets their round score regardless of bid outcome."""
        # Bidder misses, but other team should still get points
        env = self._setup_round_end(bid=9, bidder=0, round_scores=[3, 7])
        mask = torch.ones(1, dtype=torch.bool)
        env._end_round(mask)
        self.assertEqual(env.scores[0, 1].item(), 7)


class TestCheckGameEnd(unittest.TestCase):
    """Test _check_game_end() win conditions."""

    def _setup_env(self, scores, bidder, threshold=54):
        env = VectorizedPitchEnv(1, torch.device("cpu"), win_threshold=threshold)
        env.done[0] = False
        env.scores[0, 0] = scores[0]
        env.scores[0, 1] = scores[1]
        env.current_high_bidder[0] = bidder
        env.num_rounds_played[0] = 0
        return env

    def test_bidder_team_reaches_threshold(self):
        """Bidding team reaching threshold wins."""
        env = self._setup_env(scores=[54, 30], bidder=0, threshold=54)
        mask = torch.ones(1, dtype=torch.bool)
        env._check_game_end(mask)
        self.assertTrue(env.done[0].item())

    def test_non_bidder_team_at_threshold_no_win(self):
        """Non-bidding team at threshold does NOT win (bidder advantage)."""
        env = self._setup_env(scores=[54, 30], bidder=1, threshold=54)
        mask = torch.ones(1, dtype=torch.bool)
        env._check_game_end(mask)
        # score_diff = 24, which is < 54, so no win from diff either
        self.assertFalse(env.done[0].item())

    def test_score_difference_triggers_end(self):
        """Score difference >= threshold ends the game."""
        env = self._setup_env(scores=[60, -10], bidder=1, threshold=54)
        mask = torch.ones(1, dtype=torch.bool)
        env._check_game_end(mask)
        # diff = 70 >= 54
        self.assertTrue(env.done[0].item())

    def test_too_many_rounds(self):
        """Game ends after 50 rounds."""
        env = self._setup_env(scores=[10, 5], bidder=0, threshold=54)
        env.num_rounds_played[0] = 50
        mask = torch.ones(1, dtype=torch.bool)
        env._check_game_end(mask)
        self.assertTrue(env.done[0].item())

    def test_below_threshold_no_end(self):
        """Game continues when no end condition is met."""
        env = self._setup_env(scores=[30, 20], bidder=0, threshold=54)
        mask = torch.ones(1, dtype=torch.bool)
        env._check_game_end(mask)
        self.assertFalse(env.done[0].item())


class TestResetDone(unittest.TestCase):
    """Test auto-reset for completed games."""

    def test_reset_done_clears_done_flag(self):
        """After reset_done, all games should be active."""
        env = VectorizedPitchEnv(4, torch.device("cpu"))
        env.reset_all()
        env.done[1] = True
        env.done[3] = True
        env.reset_done()
        self.assertFalse(env.done.any())

    def test_reset_done_preserves_active_games(self):
        """Active games should not be affected by reset_done."""
        env = VectorizedPitchEnv(4, torch.device("cpu"))
        env.reset_all()
        # Set some scores on game 0 (active)
        env.scores[0, 0] = 15
        env.done[2] = True
        env.reset_done()
        # Game 0 scores should be unchanged
        self.assertEqual(env.scores[0, 0].item(), 15)
        # Game 2 scores should be reset
        self.assertEqual(env.scores[2, 0].item(), 0)

    def test_reset_done_new_dealer(self):
        """Reset games get a new random dealer and hands."""
        env = VectorizedPitchEnv(2, torch.device("cpu"))
        env.reset_all()
        env.done[0] = True
        env.hands[0] = 0  # clear hands
        env.reset_done()
        # Hands should be dealt again (non-zero)
        cards = (env.hands[0] != 0).sum().item()
        self.assertEqual(cards, 36)  # 4 players * 9 cards

    def test_step_ignores_done_games(self):
        """Done games should not have their state modified by step()."""
        env = VectorizedPitchEnv(2, torch.device("cpu"))
        env.reset_all()
        env.done[1] = True
        scores_before = env.scores[1].clone()
        # Step with action=0 for both games
        env.step(torch.tensor([10, 0], dtype=torch.long))
        # Done game's scores should be unchanged
        self.assertTrue(torch.equal(env.scores[1], scores_before))


class TestResetRound(unittest.TestCase):
    """Test _reset_round() preserves scores and advances dealer."""

    def test_dealer_advances(self):
        """Dealer should advance by 1 on new round."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        old_dealer = env.dealer[0].item()
        mask = torch.ones(1, dtype=torch.bool)
        env._reset_round(mask)
        new_dealer = env.dealer[0].item()
        self.assertEqual(new_dealer, (old_dealer + 1) % 4)

    def test_scores_preserved(self):
        """Scores should be preserved across round resets."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        env.scores[0, 0] = 15
        env.scores[0, 1] = -7
        mask = torch.ones(1, dtype=torch.bool)
        env._reset_round(mask)
        self.assertEqual(env.scores[0, 0].item(), 15)
        self.assertEqual(env.scores[0, 1].item(), -7)

    def test_round_count_increments(self):
        """num_rounds_played should increment."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        self.assertEqual(env.num_rounds_played[0].item(), 0)
        mask = torch.ones(1, dtype=torch.bool)
        env._reset_round(mask)
        self.assertEqual(env.num_rounds_played[0].item(), 1)

    def test_phase_resets_to_bidding(self):
        """Phase should go back to BIDDING after round reset."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        env.phase[0] = PHASE_PLAYING
        mask = torch.ones(1, dtype=torch.bool)
        env._reset_round(mask)
        self.assertEqual(env.phase[0].item(), PHASE_BIDDING)


class TestCalculateRewards(unittest.TestCase):
    """Test _calculate_rewards()."""

    def _setup_playing_env(self, trump_suit=0):
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.done[0] = False
        env.phase[0] = PHASE_PLAYING
        env.trump_suit[0] = trump_suit
        env.current_player[0] = 0
        env.playing_iterator[0] = 0
        env.scores[0] = 0
        env.round_scores[0] = 0
        return env

    def test_rewards_shape(self):
        """step() should return (N, 2) rewards for both teams."""
        env = self._setup_playing_env()
        env.hands[0, 0, 0] = encode_card(0, 15)  # Hearts Ace
        obs, rewards, dones = env.step(torch.tensor([0], dtype=torch.long))
        self.assertEqual(rewards.shape, (1, 2))

    def test_trick_points_reward(self):
        """Trick points produce opposite rewards for each team."""
        env = self._setup_playing_env()
        env.hands[0, 0, 0] = encode_card(0, 15)  # Hearts Ace (1pt)
        env.hands[0, 1, 0] = encode_card(0, 5)
        env.hands[0, 2, 0] = encode_card(0, 6)
        env.hands[0, 3, 0] = encode_card(0, 7)

        # Play all 4 cards to complete the trick
        for i in range(4):
            obs, rewards, dones = env.step(torch.tensor([0], dtype=torch.long))

        # After trick completion, team 0 won 1pt (Ace)
        # rewards[:, 0] should be positive, rewards[:, 1] negative
        self.assertEqual(rewards[0, 0].item(), -rewards[0, 1].item())

    def test_game_end_win_bonus(self):
        """Winning team gets +100 bonus, losing team gets -100."""
        env = VectorizedPitchEnv(1, torch.device("cpu"), win_threshold=5)
        env.done[0] = False
        env.phase[0] = PHASE_PLAYING
        env.trump_suit[0] = 0
        env.current_player[0] = 0
        env.current_bid[0] = 5
        env.current_high_bidder[0] = 0
        env.scores[0, 0] = 4  # Team 0 almost at threshold
        env.round_scores[0, 0] = 6  # Enough to make bid
        env.round_scores[0, 1] = 1

        # No cards → round ends
        env.playing_iterator[0] = 0
        env.hands[0] = 0

        obs, rewards, dones = env.step(torch.tensor([23], dtype=torch.long))
        self.assertTrue(env.done[0].item())
        # Team 0 wins → +100, team 1 loses → -100
        self.assertGreater(rewards[0, 0].item(), 0)
        self.assertLess(rewards[0, 1].item(), 0)

    def test_round_end_no_double_counting(self):
        """At round end, trick points already given shouldn't be re-counted."""
        env = VectorizedPitchEnv(1, torch.device("cpu"), win_threshold=54)
        env.done[0] = False
        env.phase[0] = PHASE_PLAYING
        env.trump_suit[0] = 0
        env.current_player[0] = 0
        env.playing_iterator[0] = 0
        env.current_bid[0] = 5
        env.current_high_bidder[0] = 0  # Team 0 bid 5

        # Set up: team 0 has earned 5 round_scores, team 1 has 2
        env.round_scores[0, 0] = 5
        env.round_scores[0, 1] = 2
        env.hands[0] = 0  # No cards → round will end

        obs, rewards, dones = env.step(torch.tensor([23], dtype=torch.long))

        # Team 0 made their bid (5 >= 5) → score += 5
        # Team 1 gets their round_score → score += 2
        # But those points were already rewarded per-trick during the round.
        # Round-end adjustment should be 0 for both teams (no set, no moon).
        # With no trick this step (last_trick_points=0) and adjustment=0,
        # reward should be 0 for both teams (ignoring game-end bonus).
        if not dones[0]:
            # Game didn't end, so no game-end bonus to confuse things
            self.assertAlmostEqual(rewards[0, 0].item(), 0.0, places=5)
            self.assertAlmostEqual(rewards[0, 1].item(), 0.0, places=5)

    def test_round_end_set_penalty_adjustment(self):
        """When bidder gets set, round-end adjustment reflects the penalty."""
        env = VectorizedPitchEnv(1, torch.device("cpu"), win_threshold=54)
        env.done[0] = False
        env.phase[0] = PHASE_PLAYING
        env.trump_suit[0] = 0
        env.current_player[0] = 0
        env.playing_iterator[0] = 0
        env.current_bid[0] = 7  # Bid 7
        env.current_high_bidder[0] = 0  # Team 0 bid

        # Team 0 only scored 3 (less than 7 bid) → will get set
        env.round_scores[0, 0] = 3
        env.round_scores[0, 1] = 4
        env.hands[0] = 0  # No cards → round ends

        obs, rewards, dones = env.step(torch.tensor([23], dtype=torch.long))

        # Bidder team 0 gets set: actual score_delta = -7
        # Saved round_scores for team 0 = 3
        # Adjustment for team 0 = -7 - 3 = -10
        # Team 1: actual score_delta = +4, saved = 4, adjustment = 0
        # Net reward for team 0 = 0 (trick) + (-10 - 0) = -10
        # Net reward for team 1 = 0 (trick) + (0 - (-10)) = +10
        if not dones[0]:
            self.assertAlmostEqual(rewards[0, 0].item(), -10.0, places=5)
            self.assertAlmostEqual(rewards[0, 1].item(), 10.0, places=5)


class TestDealerForcedBid(unittest.TestCase):
    """Test that dealer cannot pass when no one has bid."""

    def test_dealer_must_bid_when_all_pass(self):
        """When all other players pass, dealer can't pass (must bid minimum)."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        # Pass for first 3 players
        for _ in range(3):
            env._handle_bid(torch.tensor([10], dtype=torch.int8))

        # Now dealer's turn with current_bid=0
        self.assertEqual(env.current_player[0].item(), env.dealer[0].item())
        mask = env._get_action_mask()[0]
        # Dealer should NOT be able to pass
        self.assertEqual(mask[10].item(), 0, "Dealer should not be able to pass with no bids")
        # Dealer should be able to bid
        self.assertEqual(mask[11].item(), 1, "Dealer should be able to bid 5")


class TestMultiEnvStep(unittest.TestCase):
    """Test multi-env stepping with games in different phases."""

    def test_different_phases_simultaneously(self):
        """Games in different phases should each process correctly."""
        env = VectorizedPitchEnv(2, torch.device("cpu"))
        env.reset_all()

        # Play game 0 through bidding, keep game 1 in bidding
        # Game 0: bid, 3 passes
        env.phase[0] = PHASE_PLAYING
        env.trump_suit[0] = 0
        env.current_bid[0] = 5
        env.current_high_bidder[0] = 0
        env.current_player[0] = 0
        # Give game 0 valid cards
        env.hands[0, 0, 0] = encode_card(0, 5)

        # Game 1 is still in bidding
        self.assertEqual(env.phase[1].item(), PHASE_BIDDING)

        # Step: game 0 plays card (action 0), game 1 passes (action 10)
        masks = env._get_action_mask()
        self.assertEqual(masks[0, 0].item(), 1)  # game 0 can play slot 0
        self.assertEqual(masks[1, 10].item(), 1)  # game 1 can pass

        obs, rewards, dones = env.step(torch.tensor([0, 10], dtype=torch.long))
        # Both should still be alive
        self.assertFalse(dones[0].item())
        self.assertFalse(dones[1].item())


class TestObservationLayoutMidGame(unittest.TestCase):
    """Test observation layout matches Python env at various game states."""

    def test_obs_after_bidding(self):
        """Observation should match after bidding phase."""
        py_env, vec_env = setup_synced_envs(seed=99)
        py_obs = py_env._get_observation()

        # Make a bid
        action = 13  # bid 7
        py_obs, _, _, _, _ = py_env.step(action, py_obs)
        vec_env.step(torch.tensor([action], dtype=torch.long))

        # Sync and compare observations
        py_flat = flatten_observation(py_obs)
        vec_obs, _ = vec_env.get_observations()
        vec_flat = vec_obs[0].numpy()

        np.testing.assert_array_almost_equal(
            py_flat, vec_flat, decimal=4,
            err_msg="Observation mismatch after bid"
        )


class TestHandRemoval(unittest.TestCase):
    """Test vectorized gather-based hand removal in _handle_play."""

    def _setup_playing_env(self, hand_cards, trump_suit=0):
        """Create env with one game in PLAYING phase, player 0's hand set to hand_cards."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.done[0] = False
        env.phase[0] = PHASE_PLAYING
        env.trump_suit[0] = trump_suit
        env.current_player[0] = 0
        env.playing_iterator[0] = 0
        env.hands[0] = 0
        for i, c in enumerate(hand_cards):
            env.hands[0, 0, i] = c
        return env

    def test_remove_first_slot(self):
        """Removing slot 0 shifts all cards left."""
        cards = [encode_card(0, 2), encode_card(0, 3), encode_card(0, 5), 0, 0, 0, 0, 0, 0, 0]
        env = self._setup_playing_env(cards)
        env.step(torch.tensor([0], dtype=torch.long))  # play slot 0
        hand = env.hands[0, 0].tolist()
        self.assertEqual(hand[0], encode_card(0, 3))
        self.assertEqual(hand[1], encode_card(0, 5))
        self.assertEqual(hand[2], 0)

    def test_remove_middle_slot(self):
        """Removing a middle slot shifts only cards after it."""
        cards = [encode_card(0, 2), encode_card(0, 3), encode_card(0, 5),
                 encode_card(0, 7), 0, 0, 0, 0, 0, 0]
        env = self._setup_playing_env(cards)
        env.step(torch.tensor([1], dtype=torch.long))  # play slot 1
        hand = env.hands[0, 0].tolist()
        self.assertEqual(hand[0], encode_card(0, 2))
        self.assertEqual(hand[1], encode_card(0, 5))
        self.assertEqual(hand[2], encode_card(0, 7))
        self.assertEqual(hand[3], 0)

    def test_remove_last_occupied_slot(self):
        """Removing the last card leaves all zeros after it."""
        cards = [encode_card(0, 2), encode_card(0, 3), 0, 0, 0, 0, 0, 0, 0, 0]
        env = self._setup_playing_env(cards)
        env.step(torch.tensor([1], dtype=torch.long))  # play slot 1
        hand = env.hands[0, 0].tolist()
        self.assertEqual(hand[0], encode_card(0, 2))
        for i in range(1, 10):
            self.assertEqual(hand[i], 0)

    def test_remove_single_card(self):
        """Hand with one card → all empty after removal."""
        cards = [encode_card(0, 15), 0, 0, 0, 0, 0, 0, 0, 0, 0]
        env = self._setup_playing_env(cards)
        env.step(torch.tensor([0], dtype=torch.long))  # play slot 0
        hand = env.hands[0, 0].tolist()
        self.assertTrue(all(c == 0 for c in hand))

    def test_remove_from_full_hand(self):
        """Full 10-card hand, remove slot 5."""
        cards = [encode_card(0, r) for r in [2, 3, 5, 6, 7, 8, 9, 10, 12, 15]]
        env = self._setup_playing_env(cards)
        env.step(torch.tensor([5], dtype=torch.long))  # play slot 5 (rank 8)
        hand = env.hands[0, 0].tolist()
        expected = [encode_card(0, r) for r in [2, 3, 5, 6, 7, 9, 10, 12, 15]] + [0]
        self.assertEqual(hand, expected)


class TestNoValidPlayVectorized(unittest.TestCase):
    """Test the vectorized _handle_no_valid_play path."""

    def test_no_valid_plays_ends_round(self):
        """When no player has any valid trump card, round should end."""
        env = VectorizedPitchEnv(1, torch.device("cpu"), win_threshold=54)
        env.done[0] = False
        env.phase[0] = PHASE_PLAYING
        env.trump_suit[0] = 0  # Hearts
        env.current_player[0] = 0
        env.playing_iterator[0] = 0
        env.current_bid[0] = 5
        env.current_high_bidder[0] = 0
        # All hands empty → no valid plays
        env.hands[0] = 0
        env._create_and_shuffle_decks(torch.ones(1, dtype=torch.bool))

        phase_before = env.phase[0].item()
        env.step(torch.tensor([23], dtype=torch.long))
        # Round should have ended → back to bidding
        self.assertEqual(env.phase[0].item(), PHASE_BIDDING)

    def test_no_valid_play_mid_trick_does_not_end_round(self):
        """Action 23 mid-trick (playing_iterator > 0) should NOT end round."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.done[0] = False
        env.phase[0] = PHASE_PLAYING
        env.trump_suit[0] = 0
        env.current_player[0] = 1
        env.playing_iterator[0] = 1  # mid-trick
        env.hands[0] = 0  # no cards
        env.current_bid[0] = 5
        env.current_high_bidder[0] = 0

        env.step(torch.tensor([23], dtype=torch.long))
        # Should still be in playing phase (round not ended)
        self.assertEqual(env.phase[0].item(), PHASE_PLAYING)

    def test_has_valid_plays_does_not_end_round(self):
        """If any player has a valid card, round should not end on action 23."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.done[0] = False
        env.phase[0] = PHASE_PLAYING
        env.trump_suit[0] = 0  # Hearts
        env.current_player[0] = 0
        env.playing_iterator[0] = 0
        env.current_bid[0] = 5
        env.current_high_bidder[0] = 0
        env.hands[0] = 0
        # Player 2 has a trump card
        env.hands[0, 2, 0] = encode_card(0, 5)  # Hearts 5
        env._create_and_shuffle_decks(torch.ones(1, dtype=torch.bool))

        env.step(torch.tensor([23], dtype=torch.long))
        # Still playing — round should NOT have ended
        self.assertEqual(env.phase[0].item(), PHASE_PLAYING)


class TestDoubleMoonBid(unittest.TestCase):
    """Test action 18 (double shoot moon) bidding edge cases."""

    def test_dealer_can_double_moon_after_moon_bid(self):
        """Dealer should have action 18 available when someone bid moon."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        dealer = env.dealer[0].item()
        # Non-dealer bids moon (action 17)
        first_bidder = (dealer + 1) % 4
        env.current_player[0] = first_bidder
        env._handle_bid(torch.tensor([17], dtype=torch.int8))  # bid moon
        # Others pass until dealer
        while env.current_player[0].item() != dealer:
            env._handle_bid(torch.tensor([10], dtype=torch.int8))  # pass
        # Now dealer's turn — should be able to double moon
        mask = env._get_action_mask()[0]
        self.assertEqual(mask[18].item(), 1, "Dealer should be able to double moon")

    def test_non_dealer_cannot_double_moon(self):
        """Non-dealer should never have action 18 available."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        dealer = env.dealer[0].item()
        # First player bids moon
        env._handle_bid(torch.tensor([17], dtype=torch.int8))
        # Next player — not dealer (unless dealer is player+1, so skip if so)
        if env.current_player[0].item() != dealer:
            mask = env._get_action_mask()[0]
            self.assertEqual(mask[18].item(), 0,
                             "Non-dealer should not be able to double moon")

    def test_double_moon_sets_bid_12(self):
        """Action 18 should set current_bid to 12."""
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.reset_all()
        dealer = env.dealer[0].item()
        # Get to dealer with moon bid
        first_bidder = (dealer + 1) % 4
        env.current_player[0] = first_bidder
        env._handle_bid(torch.tensor([17], dtype=torch.int8))
        while env.current_player[0].item() != dealer:
            env._handle_bid(torch.tensor([10], dtype=torch.int8))
        # Dealer doubles
        env._handle_bid(torch.tensor([18], dtype=torch.int8))
        self.assertEqual(env.current_bid[0].item(), 12)
        self.assertEqual(env.phase[0].item(), PHASE_CHOOSESUIT)


class TestPERBufferPreAllocated(unittest.TestCase):
    """Test the pre-allocated PrioritizedReplayBuffer."""

    def test_add_and_sample_roundtrip(self):
        """Added transitions should be retrievable via sample."""
        from train import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=4, alpha=0.6)
        state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        next_state = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        buf.add(state, 2, 1.5, next_state, False)
        buf.add(state * 2, 3, -1.0, next_state * 2, True)

        self.assertEqual(buf.size, 2)
        states, actions, rewards, next_states, dones, indices, weights = buf.sample(2, beta=0.4)
        self.assertEqual(states.shape, (2, 4))
        self.assertEqual(actions.shape, (2,))
        # All sampled actions should be 2 or 3
        self.assertTrue(set(actions.tolist()).issubset({2, 3}))

    def test_circular_overwrite(self):
        """Buffer should overwrite oldest entries when full."""
        from train import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=3, obs_dim=2, alpha=0.6)
        for i in range(5):
            buf.add(np.array([float(i), 0.0]), i, 0.0, np.array([0.0, 0.0]), False)
        self.assertEqual(buf.size, 3)
        # Oldest entries (0, 1) should be overwritten by (3, 4)
        # Positions: 0→3, 1→4, 2→2
        self.assertEqual(buf.actions[0], 3)
        self.assertEqual(buf.actions[1], 4)
        self.assertEqual(buf.actions[2], 2)

    def test_sample_returns_copies(self):
        """Sampled arrays should be independent copies, not views."""
        from train import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=10, obs_dim=4, alpha=0.6)
        for i in range(5):
            buf.add(np.ones(4) * i, i, 0.0, np.zeros(4), False)
        states, *_ = buf.sample(3, beta=0.4)
        # Mutating the sample should not affect the buffer
        states[:] = 999.0
        self.assertFalse(np.any(buf.states == 999.0))

    def test_priority_update(self):
        """update_priorities should change sampling distribution."""
        from train import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=2, alpha=0.6)
        for i in range(10):
            buf.add(np.array([float(i), 0.0]), i, 0.0, np.zeros(2), False)
        # Get initial indices
        _, _, _, _, _, indices, _ = buf.sample(5, beta=0.4)
        # Boost priority of one entry
        td_errors = np.zeros(5)
        td_errors[0] = 100.0  # huge error → high priority
        buf.update_priorities(indices, td_errors)
        # That entry's priority should now be high in the tree
        boosted_idx = indices[0]
        self.assertGreater(buf.tree.tree[boosted_idx], 1.0)


class TestTrickScoringEdgeCases(unittest.TestCase):
    """Test trick scoring for point cards, off-jack, and edge cases."""

    def _setup_trick(self, trump_suit=0):
        env = VectorizedPitchEnv(1, torch.device("cpu"))
        env.done[0] = False
        env.phase[0] = PHASE_PLAYING
        env.trump_suit[0] = trump_suit
        env.current_player[0] = 0
        env.playing_iterator[0] = 0
        env.scores[0] = 0
        env.round_scores[0] = 0
        return env

    def test_three_of_trump_scores_3_points(self):
        """3-of-trump is worth 3 points (highest single-card value)."""
        env = self._setup_trick(trump_suit=0)
        env.hands[0, 0, 0] = encode_card(0, 15)  # Ace (1pt, wins)
        env.hands[0, 1, 0] = encode_card(0, 3)   # Three (3pt)
        env.hands[0, 2, 0] = encode_card(0, 5)   # 5 (0pt)
        env.hands[0, 3, 0] = encode_card(0, 6)   # 6 (0pt)

        for _ in range(4):
            env.step(torch.tensor([0], dtype=torch.long))

        self.assertEqual(env.trick_winner[0].item(), 0)  # Ace wins
        # Team 0 (trick winner): Ace(1) + Three(3) = 4 points
        self.assertEqual(env.round_scores[0, 0].item(), 4)

    def test_all_point_cards_in_one_trick(self):
        """Ace + Jack + 10 + 3 = 1+1+1+3 = 6 points in one trick."""
        env = self._setup_trick(trump_suit=0)
        env.hands[0, 0, 0] = encode_card(0, 15)  # Ace (1pt)
        env.hands[0, 1, 0] = encode_card(0, 12)  # Jack (1pt)
        env.hands[0, 2, 0] = encode_card(0, 10)  # Ten (1pt)
        env.hands[0, 3, 0] = encode_card(0, 3)   # Three (3pt)

        for _ in range(4):
            env.step(torch.tensor([0], dtype=torch.long))

        self.assertEqual(env.trick_winner[0].item(), 0)  # Ace wins
        # All points to team 0 (Ace's team): 1+1+1+3 = 6
        self.assertEqual(env.round_scores[0, 0].item(), 6)

    def test_off_jack_point_goes_to_trick_winner(self):
        """Off-jack (rank 12) scores 1 point for trick winner's team."""
        env = self._setup_trick(trump_suit=0)  # Hearts trump
        # Off-jack = Diamonds Jack (suit 1, rank 12)
        env.hands[0, 0, 0] = encode_card(0, 15)  # Hearts Ace (wins, 1pt)
        env.hands[0, 1, 0] = encode_card(1, 12)  # Off-jack (1pt)
        env.hands[0, 2, 0] = encode_card(0, 5)   # 5 (0pt)
        env.hands[0, 3, 0] = encode_card(0, 6)   # 6 (0pt)

        for _ in range(4):
            env.step(torch.tensor([0], dtype=torch.long))

        self.assertEqual(env.trick_winner[0].item(), 0)  # Ace wins
        # Ace(1) + off-jack(1) = 2 points to team 0
        self.assertEqual(env.round_scores[0, 0].item(), 2)

    def test_two_of_trump_and_three_split_scoring(self):
        """2-of-trump goes to player's team, 3-of-trump goes to trick winner."""
        env = self._setup_trick(trump_suit=0)
        # Player 1 (team 1) plays 2-of-trump, Player 0 (team 0) plays Ace and wins
        env.hands[0, 0, 0] = encode_card(0, 15)  # Ace (1pt, wins)
        env.hands[0, 1, 0] = encode_card(0, 2)   # Two (1pt → team 1)
        env.hands[0, 2, 0] = encode_card(0, 3)   # Three (3pt → trick winner)
        env.hands[0, 3, 0] = encode_card(0, 5)   # 5 (0pt)

        for _ in range(4):
            env.step(torch.tensor([0], dtype=torch.long))

        # Team 0 (winner): Ace(1) + Three(3) = 4
        # Team 1 (played 2): Two(1) = 1
        self.assertEqual(env.round_scores[0, 0].item(), 4)
        self.assertEqual(env.round_scores[0, 1].item(), 1)

    def test_trick_leader_after_resolution(self):
        """After trick resolves, current_player should be (winner + 1) % 4."""
        env = self._setup_trick(trump_suit=0)
        env.hands[0, 0, 0] = encode_card(0, 5)
        env.hands[0, 1, 0] = encode_card(0, 6)
        env.hands[0, 2, 0] = encode_card(0, 15)  # Ace → player 2 wins
        env.hands[0, 3, 0] = encode_card(0, 7)

        for _ in range(4):
            env.step(torch.tensor([0], dtype=torch.long))

        self.assertEqual(env.trick_winner[0].item(), 2)
        # Next leader = (winner + 1) % 4 = 3
        self.assertEqual(env.current_player[0].item(), 3)

    def test_two_jokers_in_trick(self):
        """Two jokers: first played should win (argmax picks lowest index on tie)."""
        env = self._setup_trick(trump_suit=0)
        env.hands[0, 0, 0] = JOKER_CODE  # Joker (rank 11, 1pt)
        env.hands[0, 1, 0] = encode_card(0, 5)   # 5 (0pt)
        env.hands[0, 2, 0] = JOKER_CODE  # Joker (rank 11, 1pt)
        env.hands[0, 3, 0] = encode_card(0, 6)   # 6 (0pt)

        for _ in range(4):
            env.step(torch.tensor([0], dtype=torch.long))

        # First joker (player 0, trick position 0) wins ties via argmax
        self.assertEqual(env.trick_winner[0].item(), 0)
        # Both jokers = 2pt to winning team (team 0)
        self.assertEqual(env.round_scores[0, 0].item(), 2)

    def test_partial_action23_trick(self):
        """1 real card + 3 action-23: real card should win the trick."""
        env = self._setup_trick(trump_suit=0)
        # Only player 0 has a card, players 1-3 have no valid cards
        env.hands[0, 0, 0] = encode_card(0, 15)  # Ace (1pt)
        env.hands[0, 1, 0] = 0
        env.hands[0, 2, 0] = 0
        env.hands[0, 3, 0] = 0
        # Give players 1-3 non-trump cards so _handle_no_valid_play doesn't end the round
        # (they have cards, just not valid ones for the mask)
        env.hands[0, 1, 1] = encode_card(2, 5)  # Clubs 5 (non-trump)
        env.hands[0, 2, 1] = encode_card(2, 6)
        env.hands[0, 3, 1] = encode_card(2, 7)

        # Player 0 plays card, players 1-3 play action 23
        env.step(torch.tensor([0], dtype=torch.long))  # player 0 plays Ace
        env.step(torch.tensor([23], dtype=torch.long))  # player 1 no valid
        env.step(torch.tensor([23], dtype=torch.long))  # player 2 no valid
        env.step(torch.tensor([23], dtype=torch.long))  # player 3 no valid

        # Player 0 (Ace) should win
        self.assertEqual(env.trick_winner[0].item(), 0)
        self.assertEqual(env.round_scores[0, 0].item(), 1)  # Ace = 1pt


class TestGameEndTiebreak(unittest.TestCase):
    """Test game-end conditions when both teams are at/above threshold."""

    def _setup_end(self, scores, bidder, threshold=54):
        env = VectorizedPitchEnv(1, torch.device("cpu"), win_threshold=threshold)
        env.done[0] = False
        env.scores[0, 0] = scores[0]
        env.scores[0, 1] = scores[1]
        env.current_high_bidder[0] = bidder
        env.num_rounds_played[0] = 0
        return env

    def test_both_at_threshold_bidder_wins(self):
        """When both teams >= threshold, bidder's team wins."""
        env = self._setup_end(scores=[55, 56], bidder=0, threshold=54)
        mask = torch.ones(1, dtype=torch.bool)
        env._check_game_end(mask)
        self.assertTrue(env.done[0].item())

    def test_both_at_threshold_non_bidder_loses(self):
        """Non-bidder team at threshold with bidder team below should not end game."""
        # Team 1 at threshold, but bidder is player 0 (team 0) who is below
        env = self._setup_end(scores=[40, 56], bidder=0, threshold=54)
        mask = torch.ones(1, dtype=torch.bool)
        env._check_game_end(mask)
        # score diff = 16 < 54, team 0 not at threshold, team 1 at threshold but not bidder
        self.assertFalse(env.done[0].item())

    def test_negative_score_difference_ends_game(self):
        """Large negative score difference (one team far behind) triggers end."""
        env = self._setup_end(scores=[-20, 40], bidder=1, threshold=54)
        mask = torch.ones(1, dtype=torch.bool)
        env._check_game_end(mask)
        # diff = |(-20) - 40| = 60 >= 54
        self.assertTrue(env.done[0].item())


class TestPERBoundary(unittest.TestCase):
    """Test PER sampling at exact buffer capacity boundaries."""

    def test_sample_at_exact_batch_size(self):
        """Sampling when size == batch_size should not crash or return index 0 for all."""
        from train import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=4, alpha=0.6)
        batch_size = 8
        for i in range(batch_size):
            buf.add(np.ones(4) * i, i, float(i), np.zeros(4), False)
        self.assertEqual(buf.size, batch_size)
        states, actions, rewards, _, _, indices, weights = buf.sample(batch_size, beta=0.4)
        # Should not all be the same index
        self.assertGreater(len(set(actions.tolist())), 1,
                           "All samples collapsed to same entry")

    def test_sample_single_entry(self):
        """Sampling from buffer with only 1 entry should return that entry."""
        from train import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=2, alpha=0.6)
        buf.add(np.array([42.0, 99.0]), 7, 3.14, np.array([1.0, 2.0]), True)
        states, actions, rewards, next_states, dones, _, _ = buf.sample(4, beta=0.4)
        # All 4 samples should be the same single entry
        self.assertTrue(np.all(actions == 7))
        self.assertTrue(np.all(np.isclose(rewards, 3.14)))
        self.assertTrue(np.all(dones))


class TestActionMaskAllPhases(unittest.TestCase):
    """Test that action masks never produce all-zeros for active games."""

    def test_mask_never_all_zero_during_full_game(self):
        """Play 50 random games, verify mask always has at least one valid action."""
        for seed in range(50):
            env = VectorizedPitchEnv(1, torch.device("cpu"))
            env.reset_all()
            rng = np.random.RandomState(seed)
            for step in range(2000):
                if env.done[0].item():
                    break
                mask = env._get_action_mask()[0].numpy()
                valid = np.where(mask == 1)[0]
                self.assertGreater(
                    len(valid), 0,
                    f"All-zero mask at seed={seed} step={step} "
                    f"phase={env.phase[0].item()} player={env.current_player[0].item()}"
                )
                action = int(rng.choice(valid))
                env.step(torch.tensor([action], dtype=torch.long))


class TestGetObservationsReturnsTuple(unittest.TestCase):
    """Test that get_observations returns (obs, masks) tuple."""

    def test_return_shape_and_content(self):
        """get_observations returns (obs, masks) with correct shapes."""
        env = VectorizedPitchEnv(4, torch.device("cpu"))
        env.reset_all()
        obs, masks = env.get_observations()
        self.assertEqual(obs.shape, (4, 119))
        self.assertEqual(masks.shape, (4, 24))
        # Masks should be int8 (raw action mask)
        self.assertEqual(masks.dtype, torch.int8)
        # Masks should match a separate _get_action_mask call
        masks_direct = env._get_action_mask()
        torch.testing.assert_close(masks, masks_direct)

    def test_obs_contains_mask_values(self):
        """The last 24 values of obs should match the returned masks."""
        env = VectorizedPitchEnv(2, torch.device("cpu"))
        env.reset_all()
        obs, masks = env.get_observations()
        # Last 24 columns of obs are the float mask
        obs_mask = obs[:, -24:]
        np.testing.assert_array_almost_equal(
            obs_mask.numpy(), masks.float().numpy(), decimal=4
        )


if __name__ == "__main__":
    unittest.main()
