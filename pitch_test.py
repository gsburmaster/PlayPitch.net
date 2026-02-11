import unittest
import numpy as np
from pitch_env import Phase, PitchEnv, Card, Suit

class TestPitchEnv(unittest.TestCase):

    def setUp(self):
        self.env = PitchEnv()

    def test_reset(self):
        obs, _ = self.env.reset()
        self.assertEqual(self.env.phase, Phase.BIDDING)
        self.assertEqual(len(self.env.hands[0]), 9)
        self.assertEqual(len(self.env.hands[1]), 9)
        self.assertEqual(len(self.env.hands[2]), 9)
        self.assertEqual(len(self.env.hands[3]), 9)
        self.assertEqual(self.env.scores, [0, 0])
        self.assertEqual(self.env.current_bid, 0)
        self.assertIsNone(self.env.trump_suit)
        self.assertIn(self.env.dealer, range(4))
        self.assertIn(self.env.current_player, range(4))

    def test_handle_bid(self):
        self.env.reset()
        self.env._handle_bid(11)  # Bid 5
        self.assertEqual(self.env.current_bid, 5)
        self.assertEqual(self.env.current_high_bidder, (self.env.current_player+3) % 4)

    def test_handle_pass(self):
        self.env.reset()
        currPlayer = self.env.current_player
        self.env._handle_bid(10)
        self.assertEqual(self.env.current_bid,0)
        self.assertEqual(self.env.current_player,(currPlayer + 1) % 4)

    def test_handle_suit_choice(self):
        self.env.reset()
        self.env.phase = Phase.CHOOSESUIT
        self.env.current_high_bidder = self.env.current_player
        self.env._handle_choose_suit(19)  # Choose Hearts
        self.assertEqual(self.env.trump_suit, Suit.HEARTS)
        self.assertEqual(self.env.phase, Phase.PLAYING)

    def test_handle_play(self):
        self.env.reset()
        self.env.phase = Phase.PLAYING
        initial_hand_size = len(self.env.hands[self.env.current_player])
        self.env._handle_play(0)  # Play first card in hand
        self.assertEqual(len(self.env.hands[(self.env.current_player+3) % 4]), initial_hand_size - 1)
        self.assertEqual(len(self.env.current_trick), 1)
        self.assertEqual(len(self.env.played_cards), 1)

    def test_handle_play_no_moves_others_can_move(self):
        self.env.reset()
        self.env.trump_suit = Suit.HEARTS
        self.env.hands[self.env.dealer] = [Card(Suit.HEARTS,3)]
        self.env.phase = Phase.PLAYING
        self.env._handle_play(23) 
        self.assertEqual(self.env.playing_iterator,1)

    def test_handle_play_only_2_cards(self):
        self.env.reset()
        self.env.phase = Phase.PLAYING
        self.env.current_trick = [
            Card(Suit.HEARTS,3)
        ]
        self.env._handle_play(23) #TODO finish
        self.assertEqual(self.env.playing_iterator,1)

    def test_resolve_trick_no_points(self):
        self.env.reset()
        self.env.trump_suit = Suit.HEARTS
        self.env.current_trick = [
            (Card(Suit.HEARTS, 4),1),
            (Card(Suit.HEARTS, 5),2),
            (Card(Suit.HEARTS, 6),3)
        ]
        self.env._resolve_trick()
        self.assertEqual(self.env.trick_winner, 3)  # 6 of hearts should win
        self.assertEqual(len(self.env.tricks), 1)
        self.assertEqual(self.env.scores[1], 0)  # 0 points     

    def test_resolve_trick_no_2(self):
        self.env.reset()
        self.env.trump_suit = Suit.HEARTS
        self.env.current_trick = [
            (Card(Suit.HEARTS, 5),0),
            (Card(Suit.HEARTS, 10),1),
            (Card(Suit.HEARTS, 4),2),
            (Card(Suit.HEARTS, 6),3)
        ]
        self.env._resolve_trick()
        self.assertEqual(self.env.trick_winner, 1)  # 10 should win
        self.assertEqual(len(self.env.tricks), 1)
        self.assertEqual(self.env.round_scores[1], 1)  # 0 points 

    def test_action_mask_hand(self):
        self.env.reset()
        self.env.trump_suit = Suit.DIAMONDS
        self.env.phase = Phase.PLAYING
        self.env.current_player = 1
        self.env.hands[1] = [Card(Suit.HEARTS, 7),Card(Suit.DIAMONDS, 8),Card(Suit.DIAMONDS, 13),Card(Suit.CLUBS, 7),Card(Suit.HEARTS, 4),Card(Suit.CLUBS, 14),Card(Suit.CLUBS, 5),Card(Suit.HEARTS, 13),Card(Suit.SPADES, 7)]
        out = self.env._get_action_mask()
        self.assertEqual(out[0:9].tolist(),[0,1,1,0,0,0,0,0,0])

    def test_resolve_trick_off_jack_wins(self):
        self.env.reset()
        self.env.trump_suit = Suit.HEARTS
        self.env.current_trick = [
            (Card(Suit.DIAMONDS, 12),0),
            (Card(Suit.HEARTS, 10),1),
            (Card(Suit.HEARTS, 4),2),
            (Card(Suit.HEARTS, 6),3)
        ]
        self.env._resolve_trick()
        self.assertEqual(self.env.trick_winner, 0)  # Off Jack should win
        self.assertEqual(len(self.env.tricks), 1)
        self.assertEqual(self.env.round_scores[0], 2)  # 0 points 

    def test_card_points(self):
        self.assertEqual(self.env._card_points(Card(Suit.HEARTS, 15)), 1)  # Ace
        self.assertEqual(self.env._card_points(Card(Suit.HEARTS, 12)), 1)  # Jack
        self.assertEqual(self.env._card_points(Card(Suit.HEARTS, 10)), 1)  # 10
        self.assertEqual(self.env._card_points(Card(Suit.HEARTS, 3)), 3)   # 3
        self.assertEqual(self.env._card_points(Card(Suit.HEARTS, 2)), 1)   # 2
        self.assertEqual(self.env._card_points(Card(None, 11)), 1)         # Joker
        self.assertEqual(self.env._card_points(Card(Suit.HEARTS, 7)), 0)   # 7 (no points)

    def test_is_valid_play(self):
        self.env.reset()
        self.env.trump_suit = Suit.HEARTS
        self.assertTrue(self.env._is_valid_play(Card(Suit.HEARTS, 7)))    # Trump suit
        self.assertTrue(self.env._is_valid_play(Card(None, 11)))          # Joker
        self.assertTrue(self.env._is_valid_play(Card(Suit.DIAMONDS, 12))) # Off Jack
        self.assertFalse(self.env._is_valid_play(Card(Suit.CLUBS, 7)))    # Non-trump, non-special card

    def test_get_action_mask_bidding(self):
        self.env.reset()
        self.env.current_bid = 0
        mask = self.env._get_action_mask()
        self.assertEqual(mask[10:19].tolist(), [1, 1, 1, 1, 1, 1, 1, 1, 0])  # Can pass, bid 5-10, shoot moon

    def test_get_action_mask_bidding_double_shoot(self):
        self.env.reset()
        self.env.current_bid = 8
        self.env.current_high_bidder = self.env.dealer
        self.env.current_player = self.env.dealer
        mask = self.env._get_action_mask()
        self.assertEqual(mask[10:19].tolist(), [1, 0, 0, 0, 0, 0, 0, 0, 1])  # Can pass, double shoot moon

    def test_get_action_mask_bidding_dealer_must_bid(self):
        self.env.reset()
        self.env.current_bid = 0
        self.env.current_high_bidder = (self.env.dealer+3) % 4
        self.env.current_player = self.env.dealer
        mask = self.env._get_action_mask()
        self.assertEqual(mask[10:19].tolist(), [0, 1, 1, 1, 1, 1, 1, 1, 0])  # Can bid 5-10, shoot moon

    def test_get_action_mask_bidding_non_dealer_can_bid(self):
        self.env.reset()
        self.env.current_bid = 6
        self.env.current_high_bidder = 0
        self.env.current_player = 1
        mask = self.env._get_action_mask()
        self.assertEqual(mask[10:19].tolist(), [1, 0, 0, 1, 1, 1, 1, 1, 0])  # Can bid 7-10, shoot moon


    def test_get_action_mask_suit_choice(self):
        self.env.reset()
        self.env.phase = Phase.CHOOSESUIT
        mask = self.env._get_action_mask()
        self.assertEqual(mask[19:23].tolist(), [1, 1, 1, 1])  # Can choose any suit

    def test_get_action_mask_playing(self):
        self.env.reset()
        self.env.phase = Phase.PLAYING
        self.env.trump_suit = Suit.HEARTS
        self.env.hands[self.env.current_player] = [Card(Suit.HEARTS, 7), Card(Suit.CLUBS, 8)]
        mask = self.env._get_action_mask()
        self.assertEqual(mask[:2].tolist(), [1, 0])  # Can play the Hearts card, but not the Clubs card

    def test_get_action_mask_playing_no_valid(self):
        self.env.reset()
        self.env.phase = Phase.PLAYING
        self.env.trump_suit = Suit.HEARTS
        self.env.hands[self.env.current_player] = [Card(Suit.SPADES, 7), Card(Suit.CLUBS, 8)]
        mask = self.env._get_action_mask()
        self.assertEqual(mask.tolist(), [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])  # Can play nothing

    def test_get_action_mask_multiple_plays_joker_first(self):
        self.env.reset()
        self.env.phase = Phase.PLAYING
        self.env.trump_suit = Suit.DIAMONDS
        self.env.hands[self.env.current_player] = [Card(Suit.HEARTS, 6), Card(Suit.HEARTS, 2), Card(Suit.SPADES, 14), Card(Suit.HEARTS, 7), Card(None, 11), Card(Suit.CLUBS, 5), Card(Suit.HEARTS, 3), Card(Suit.SPADES, 4), Card(Suit.DIAMONDS, 15)]
        mask = self.env._get_action_mask()
        self.assertEqual(mask.tolist(),[0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])


    def test_end_round_made_bid(self):
        self.env.reset()
        self.env.phase = Phase.PLAYING
        self.env.dealer = 0
        self.env.trump_suit = Suit.HEARTS
        self.env.hands[self.env.current_player] = [Card(Suit.SPADES,4),Card(Suit.DIAMONDS,9)]
        self.env.current_high_bidder = 0
        self.env.scores = [1,1]
        self.env.current_bid = 5
        self.env.round_scores = [7,3]
        self.env.number_of_rounds_played = 9
        self.env._end_round()
        self.assertEqual(self.env.phase,Phase.BIDDING)
        self.assertEqual(self.env.scores,[8,4])
        self.assertEqual(self.env.number_of_rounds_played, 10)
        self.assertEqual(self.env.current_bid,0)
        for i in range (0,4):
            self.assertEqual(len(self.env.hands[i]),9)
        self.assertEqual(self.env.trump_suit,None)
        self.assertEqual(self.env.dealer,1)
        
    def test_end_round_went_set(self):
        self.env.reset()
        self.env.phase = Phase.PLAYING
        self.env.dealer = 0
        self.env.trump_suit = Suit.HEARTS
        self.env.hands[self.env.current_player] = [Card(Suit.SPADES,4),Card(Suit.DIAMONDS,9)]
        self.env.current_high_bidder = 0
        self.env.scores = [1,1]
        self.env.current_bid = 5
        self.env.round_scores = [4,6]
        self.env.number_of_rounds_played = 9
        self.env._end_round()
        self.assertEqual(self.env.phase,Phase.BIDDING)
        self.assertEqual(self.env.scores,[-4,7])
        self.assertEqual(self.env.number_of_rounds_played, 10)
        self.assertEqual(self.env.current_bid,0)
        for i in range (0,4):
            self.assertEqual(len(self.env.hands[i]),9)
        self.assertEqual(self.env.trump_suit,None)
        self.assertEqual(self.env.dealer,1)


    # --- Moon scoring ---

    def test_end_round_shoot_moon_made(self):
        self.env.reset()
        self.env.current_high_bidder = 0
        self.env.current_bid = 11  # shoot moon
        self.env.scores = [10, 5]
        self.env.round_scores = [10, 0]
        self.env._end_round()
        self.assertEqual(self.env.scores[0], 30)  # +20 for making moon
        self.assertEqual(self.env.scores[1], 5)

    def test_end_round_shoot_moon_set(self):
        self.env.reset()
        self.env.current_high_bidder = 0
        self.env.current_bid = 11  # shoot moon
        self.env.scores = [10, 5]
        self.env.round_scores = [9, 1]
        self.env._end_round()
        self.assertEqual(self.env.scores[0], -10)  # -20 for missing moon
        self.assertEqual(self.env.scores[1], 6)

    def test_end_round_double_shoot_made(self):
        self.env.reset()
        self.env.current_high_bidder = 1
        self.env.current_bid = 12  # double shoot
        self.env.scores = [0, 0]
        self.env.round_scores = [0, 10]
        self.env._end_round()
        self.assertEqual(self.env.scores[1], 40)  # +40 for double moon
        self.assertEqual(self.env.scores[0], 0)

    def test_end_round_double_shoot_set(self):
        self.env.reset()
        self.env.current_high_bidder = 1
        self.env.current_bid = 12  # double shoot
        self.env.scores = [0, 0]
        self.env.round_scores = [2, 8]
        self.env._end_round()
        self.assertEqual(self.env.scores[1], -40)  # -40 for missing double moon
        self.assertEqual(self.env.scores[0], 2)

    # --- Reward ---

    def test_reward_zero_on_non_trick_step(self):
        self.env.reset()
        self.env.last_trick_points = [0, 0]
        reward = self.env._calculate_reward(team=0, scores_before=list(self.env.scores))
        self.assertEqual(reward, 0)

    def test_reward_trick_win(self):
        self.env.reset()
        self.env.last_trick_points = [3, 0]  # team 0 won 3 points
        reward = self.env._calculate_reward(team=0, scores_before=list(self.env.scores))
        self.assertEqual(reward, 3)

    def test_reward_trick_loss(self):
        self.env.reset()
        self.env.last_trick_points = [3, 0]  # team 0 won 3 points
        reward = self.env._calculate_reward(team=1, scores_before=list(self.env.scores))
        self.assertEqual(reward, -3)

    def test_reward_includes_set_penalty(self):
        self.env.reset()
        self.env.current_high_bidder = 0
        self.env.current_bid = 7
        self.env.scores = [10, 5]
        self.env.round_scores = [4, 6]
        scores_before = list(self.env.scores)
        self.env.last_trick_points = [0, 0]
        self.env._end_round()
        reward = self.env._calculate_reward(team=0, scores_before=scores_before)
        # Team 0 went set: 10 → 3 (-7), Team 1 banked: 5 → 11 (+6)
        # score_delta = -7 - 6 = -13
        self.assertEqual(reward, -13)

    def test_reward_game_end_bonus(self):
        self.env.reset()
        self.env.scores = [54, 0]
        self.env.current_high_bidder = 0
        self.env.current_player = 0
        self.env.last_trick_points = [0, 0]
        reward = self.env._calculate_reward(team=0, scores_before=[54, 0])
        self.assertEqual(reward, 100)

    def test_reward_game_end_loss(self):
        self.env.reset()
        self.env.scores = [54, 0]
        self.env.current_high_bidder = 0
        self.env.current_player = 1
        self.env.last_trick_points = [0, 0]
        reward = self.env._calculate_reward(team=1, scores_before=[54, 0])
        self.assertEqual(reward, -100)

    # --- Game end ---

    def test_game_not_over(self):
        self.env.reset()
        self.env.scores = [53, 0]
        self.env.current_high_bidder = 0
        self.assertFalse(self.env._check_game_end())

    def test_game_over_bidder_wins(self):
        self.env.reset()
        self.env.scores = [54, 0]
        self.env.current_high_bidder = 0
        self.assertTrue(self.env._check_game_end())

    def test_game_not_over_non_bidder_above_53(self):
        self.env.reset()
        self.env.scores = [54, 10]
        self.env.current_high_bidder = 1  # team 1 is bidder, team 0 just has points
        # team 0 has 54 but they're not the bidder, and difference is only 44
        self.assertFalse(self.env._check_game_end())

    def test_game_over_blowout(self):
        self.env.reset()
        self.env.scores = [54, 0]
        self.env.current_high_bidder = 1
        # abs(54-0) > 53, so game ends regardless of bidder
        self.assertTrue(self.env._check_game_end())

    # --- Off-jack all suit pairs ---

    def test_off_jack_clubs_trump(self):
        self.env.reset()
        self.env.trump_suit = Suit.CLUBS
        self.assertTrue(self.env._is_valid_play(Card(Suit.SPADES, 12)))
        self.assertFalse(self.env._is_valid_play(Card(Suit.HEARTS, 12)))

    def test_off_jack_spades_trump(self):
        self.env.reset()
        self.env.trump_suit = Suit.SPADES
        self.assertTrue(self.env._is_valid_play(Card(Suit.CLUBS, 12)))
        self.assertFalse(self.env._is_valid_play(Card(Suit.DIAMONDS, 12)))

    def test_off_jack_diamonds_trump(self):
        self.env.reset()
        self.env.trump_suit = Suit.DIAMONDS
        self.assertTrue(self.env._is_valid_play(Card(Suit.HEARTS, 12)))
        self.assertFalse(self.env._is_valid_play(Card(Suit.SPADES, 12)))

    # --- Observation consistency ---

    def test_observation_fixed_size(self):
        obs1, _ = self.env.reset()
        flat1 = []
        for value in obs1.values():
            if isinstance(value, np.ndarray):
                flat1.extend(value.flatten())
            elif isinstance(value, (int, np.integer)):
                flat1.append(value)
        size1 = len(flat1)

        # Play through several steps and verify size stays the same
        for _ in range(20):
            mask = obs1['action_mask']
            action = np.random.choice(np.where(mask == 1)[0])
            obs1, _, done, _, _ = self.env.step(action, obs1)
            flat = []
            for value in obs1.values():
                if isinstance(value, np.ndarray):
                    flat.extend(value.flatten())
                elif isinstance(value, (int, np.integer)):
                    flat.append(value)
            self.assertEqual(len(flat), size1, f"Observation size changed from {size1} to {len(flat)}")
            if done:
                break

    # --- Bidding round flow ---

    def test_full_bidding_round(self):
        self.env.reset()
        dealer = self.env.dealer
        first_bidder = (dealer + 1) % 4

        # First 3 players pass, each pass advances current_player
        self.env._handle_bid(11)  # player bids 5
        self.env._handle_bid(10)  # next player passes
        self.env._handle_bid(10)  # next player passes
        # Now we've reached the dealer, bidding should end
        self.assertEqual(self.env.phase, Phase.CHOOSESUIT)
        self.assertEqual(self.env.current_player, first_bidder)
        self.assertEqual(self.env.current_high_bidder, first_bidder)

    # --- Resolve trick with joker ---

    def test_resolve_trick_joker_loses_to_trump(self):
        self.env.reset()
        self.env.trump_suit = Suit.HEARTS
        self.env.current_trick = [
            (Card(None, 11), 0),         # Joker
            (Card(Suit.HEARTS, 15), 1),  # Ace of trump
        ]
        self.env._resolve_trick()
        self.assertEqual(self.env.trick_winner, 1)  # Ace beats joker by rank

    def test_resolve_trick_joker_beats_low_trump(self):
        self.env.reset()
        self.env.trump_suit = Suit.HEARTS
        self.env.current_trick = [
            (Card(None, 11), 0),        # Joker (rank 11)
            (Card(Suit.HEARTS, 4), 1),  # 4 of trump
        ]
        self.env._resolve_trick()
        self.assertEqual(self.env.trick_winner, 0)  # Joker rank 11 > 4


if __name__ == '__main__':
    unittest.main()