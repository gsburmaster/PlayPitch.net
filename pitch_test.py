import unittest
import numpy as np
from pitch_env import PitchEnv, Card, Suit

class TestPitchEnv(unittest.TestCase):

    def setUp(self):
        self.env = PitchEnv()

    def test_reset(self):
        obs, _ = self.env.reset()
        self.assertEqual(self.env.phase, 0)
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

    def test_handle_suit_choice(self):
        self.env.reset()
        self.env.phase = 1
        self.env.current_high_bidder = self.env.current_player
        self.env._handle_choose_suit(19)  # Choose Hearts
        self.assertEqual(self.env.trump_suit, 0)
        self.assertEqual(self.env.phase, 2)

    def test_handle_play(self):
        self.env.reset()
        self.env.phase = 2
        initial_hand_size = len(self.env.hands[self.env.current_player])
        self.env._handle_play(0)  # Play first card in hand
        self.assertEqual(len(self.env.hands[(self.env.current_player+3) % 4]), initial_hand_size - 1)
        self.assertEqual(len(self.env.current_trick), 1)
        self.assertEqual(len(self.env.played_cards), 1)

    def test_handle_play_no_moves(self):
        self.env.reset()
        self.env.phase = 2
        self.env._handle_play(23) 
        self.assertEqual(self.env.playing_iterator,1)

    def test_handle_play_only_2_cards(self):
        self.env.reset()
        self.env.phase = 2
        self.env.current_trick = [
            Card(Suit.HEARTS,3)
        ]
        self.env._handle_play(23) #TODO finish
        self.assertEqual(self.env.playing_iterator,1)

    def test_resolve_trick_no_points(self):
        self.env.reset()
        self.env.trump_suit = Suit.HEARTS
        self.env.current_trick = [
            Card(Suit.HEARTS, 7),
            Card(Suit.HEARTS, 4),
            Card(Suit.HEARTS, 5),
            Card(Suit.HEARTS, 6)
        ]
        self.env._resolve_trick()
        self.assertEqual(self.env.trick_winner, 0)  # 7 should win
        self.assertEqual(len(self.env.tricks), 1)
        self.assertEqual(self.env.scores[1], 0)  # 0 points 

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
        self.assertEqual(self.env.scores[1], 1)  # 0 points 


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
        self.assertEqual(self.env.scores[0], 2)  # 0 points 

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
        self.env.phase = 1
        mask = self.env._get_action_mask()
        self.assertEqual(mask[19:23].tolist(), [1, 1, 1, 1])  # Can choose any suit

    def test_get_action_mask_playing(self):
        self.env.reset()
        self.env.phase = 2
        self.env.trump_suit = Suit.HEARTS
        self.env.hands[self.env.current_player] = [Card(Suit.HEARTS, 7), Card(Suit.CLUBS, 8)]
        mask = self.env._get_action_mask()
        self.assertEqual(mask[:2].tolist(), [1, 0])  # Can play the Hearts card, but not the Clubs card

    def test_get_action_mask_playing_no_valid(self):
        self.env.reset()
        self.env.phase = 2
        self.env.trump_suit = Suit.HEARTS
        self.env.hands[self.env.current_player] = [Card(Suit.SPADES, 7), Card(Suit.CLUBS, 8)]
        mask = self.env._get_action_mask()
        self.assertEqual(mask.tolist(), [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])  # Can play nothing


if __name__ == '__main__':
    unittest.main()