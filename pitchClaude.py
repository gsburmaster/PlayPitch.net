import gymnasium as gym
import numpy as np
from enum import Enum
from typing import List, Tuple, Dict

class Suit(Enum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3

class Card:
    def __init__(self, suit: Suit, rank: int):
        self.suit = suit
        self.rank = rank  # 2-10 for number cards, 11 for Joker, 12-15 for J,Q,K,A

    def __str__(self):
        ranks = {11: 'Joker', 12: 'J', 13: 'Q', 14: 'K', 15: 'A'}
        return f"{ranks.get(self.rank, self.rank)} of {self.suit.name if self.suit else ''}"

    def __lt__(self, other):
        return self.rank < other.rank

class PitchEnv(gym.Env):
    def __init__(self):
        super(PitchEnv, self).__init__()
        self.played_cards = []
        self.num_actions = 22  # 9 for cards in hand, 9 for possible bids (pass, 5-10, shoot the moon, double shoot the moon), 4 for choosing suit
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Dict({
            'hand': gym.spaces.Box(low=0, high=15, shape=(9, 2), dtype=np.int8),
            'tricks': gym.spaces.Box(low=0, high=15, shape=(4, 2), dtype=np.int8),
            'played_cards': gym.spaces.Box(low=0, high=15, shape=(16, 2), dtype=np.int8),
            'scores': gym.spaces.Box(low=0, high=54, shape=(2,), dtype=np.int8),
            'current_bid': gym.spaces.Discrete(9),  # 0,5-10,moon,double moon where 0 means no bid yet
            'current_bidder': gym.spaces.Discrete(4),
            'dealer': gym.spaces.Discrete(4),
            'current_player': gym.spaces.Discrete(4),
            'trump_suit': gym.spaces.Discrete(5),  # 0-3 for suits, 4 for no trump
            'phase': gym.spaces.Discrete(3),  # 0: bidding, 1: choosing suit, 2: playing
            'action_mask': gym.spaces.Box(low=0, high=1, shape=(self.num_actions,), dtype=np.int8)
        })
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.deck = self._create_deck()
        self.hands = [[] for _ in range(4)]
        self.scores = [0, 0]
        self.current_bid = 0
        self.current_bidder = 0
        self.dealer = self.np_random.integers(4)
        self.current_player = (self.dealer + 1) % 4
        self.trump_suit = None
        self.phase = 0  # 0: bidding, 1: playing
        self.tricks = []
        self.played_cards = []
        self.current_trick = []
        self.trick_winner = None

        self._deal_cards()
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        observation = self._get_observation()
        if observation['action_mask'][action] == 0:
            valid_actions = np.where(observation['action_mask'] == 1)[0]
            action = self.np_random.choice(valid_actions)

        if self.phase == 0:  # Bidding phase
            self._handle_bid(action)
        if self.phase == 1:
            self._handle_suit_choice(action)
        elif self.phase == 2:  # Playing phase
            self._handle_play(action)

        terminated = self._check_game_end()
        truncated = False
        reward = self._calculate_reward()
        observation = self._get_observation()
        info = {}

        return observation, reward, terminated, truncated, info

    def _create_deck(self):
        deck = [Card(suit, rank) for suit in Suit for rank in range(2, 11)] + [Card(suit, rank) for suit in Suit for rank in range(12, 16)] #does this have 11 rank in it?
        deck.extend([Card(None, 11), Card(None, 11)])  # Add two jokers
        self.np_random.shuffle(deck)
        return deck

    def _deal_cards(self):
        for _ in range(9):
            for player in range(4):
                self.hands[player].append(self.deck.pop())

    def _handle_bid(self, action):
        if 9 <= action <= 14: #TODO fix this
            self.current_bid = action - 4  # Convert to bid value (5-10)
        self.current_player = (self.current_player + 1) % 4
        if self.current_player == (self.dealer + 1) % 4:
            self.phase = 1  # Move to playing phase

    def _handle_choose_suit(self,action):
        
        pass

    def _handle_play(self, action):
        card = self.hands[self.current_player][action]
        self.current_trick.append(card)
        self.hands[self.current_player].remove(card)
        self.played_cards.append(card)
        if len(self.current_trick) == 4:
            self._resolve_trick()

        self.current_player = (self.current_player + 1) % 4

    def _discard_and_fill(self):
        for player in range(4):
            self.hands[player] = [card for card in self.hands[player]
                                  if card.suit == self.trump_suit or card.rank == 11]
            while len(self.hands[player]) < 6 and self.deck:
                self.hands[player].append(self.deck.pop())

    def _resolve_trick(self):
        winning_card = max(self.current_trick, key=lambda c: (c.suit == self.trump_suit, c.rank))
        self.trick_winner = self.current_trick.index(winning_card)
        self.tricks.append(self.current_trick)
        self.current_trick = []
        self.current_player = self.trick_winner

        trick_points = sum(self._card_points(card) for card in self.tricks[-1])
        self.scores[self.trick_winner % 2] += trick_points

    def _card_points(self, card):
        if card.rank in [15, 12, 10]:  # Ace, Jack, 10
            return 1
        elif card.rank == 11:  # Joker
            return 1
        elif card.rank == 3:
            return 3
        elif card.rank == 2:
            return 1
        return 0

    def _check_game_end(self):
        return len(self.tricks) == 6

    def _calculate_reward(self):
        # Implement reward calculation based on game state
        return 0

    def _get_observation(self):
        return {
            'hand': np.array([(card.suit.value if card.suit else 4, card.rank) for card in self.hands[self.current_player]]),
            'played_cards': np.array([(card.suit.value if card.suit else 4, card.rank) for card in self.played_cards]),
            'tricks': np.array([(card.suit.value if card.suit else 4, card.rank) for card in self.current_trick]),
            'scores': np.array(self.scores),
            'current_bid': self.current_bid,
            'current_bidder': self.current_bidder,
            'dealer': self.dealer,
            'current_player': self.current_player,
            'trump_suit': self.trump_suit.value if self.trump_suit else 4,
            'phase': self.phase,
            'action_mask': self._get_action_mask()
        }

    def _get_action_mask(self):
        mask = np.zeros(self.num_actions, dtype=np.int8)
        
        if self.phase == 0:  # Bidding phase
            min = self.current_bid + 9
            if min < 15:
                mask[min+1:16] = 1  # Allow bids from 5 to 10, or pass (index 9 to 16)
            elif min == 15:
                mask[16] = 1  
            if not (self.current_bid != 0 and self.current_player == self.dealer):
                mask[9] = 1
            if (self.current_player == self.dealer and self.current_bid == 7):
                mask[17] = 1 # double shoot if someone has shot already and current player is dealer
        elif self.phase == 1: # suit selection phase
            mask[18:21] = 1
        elif self.phase == 2:  # Playing phase
            for i, card in enumerate(self.hands[self.current_player]):
                if self._is_valid_play(card):
                    mask[i] = 1
        
        return mask
    def _is_off_jack(self,card):
        if card.rank != 12:
            return False
        switch = {
            Suit.CLUBS: card.suit == Suit.SPADES,
            Suit.SPADES: card.suit == Suit.CLUBS,
            Suit.DIAMONDS: card.suit == Suit.HEARTS,
            Suit.HEARTS: card.suit == Suit.DIAMONDS
        }
        return switch.get(self.trump_suit)

    def _is_valid_play(self, card as Card):
        if (card.suit == self.trump_suit or card.value == 11 or self._is_off_jack(card)):
            return True
        # Implement logic to check if a card is valid to play
        # based on the current trick and game rules
        return False  # Placeholder

# Example usage:
env = PitchEnv()
obs, info = env.reset()
terminated = truncated = False
while not terminated and not truncated:
    action = env.action_space.sample()  # Replace with your agent's action
    obs, reward, terminated, truncated, info = env.step(action)