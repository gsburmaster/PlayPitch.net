import sys
import gymnasium as gym
import numpy as np
from enum import Enum
from typing import List, Tuple, Dict


#TODO LIST
# - fix mask numbers
# figure out how to end the 'playing phase'
# write unit tests for discard and fill
# fix discard and fill
# add observations for how many cards each player takes
# figure out how to end each trick when no valid moves
# assume burn worst cards-always the right play 

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
        self.num_actions = 24  # 10 for cards in hand, 9 for possible bids (pass, 5-10, shoot the moon, double shoot the moon), 4 for choosing suit, one for no valid play
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Dict({
            'hand': gym.spaces.Box(low=0, high=15, shape=(10, 2), dtype=np.int8), # possible to have up to ten cards
            'tricks': gym.spaces.Box(low=0, high=15, shape=(4, 2), dtype=np.int8),
            'played_cards': gym.spaces.Box(low=0, high=15, shape=(16, 2), dtype=np.int8),
            'scores': gym.spaces.Box(low=0, high=54, shape=(2,), dtype=np.int8),
            'current_bid': gym.spaces.Discrete(9),  # 0,5-10,moon,double moon where 0 means no bid yet
            'current_high_bidder': gym.spaces.Discrete(5), #no bid or 1-4
            'dealer': gym.spaces.Discrete(4),
            'number_of_rounds_played': gym.spaces.Box(low=0,high=2**63 - 2, dtype=np.uint64), # how many rounds has the game gone
            'current_player': gym.spaces.Discrete(4),
            'player_cards_taken': gym.spaces.Box(low=-1,high=10, shape=(4,),dtype=np.int8),
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
        self.current_high_bidder = 0
        self.dealer = self.np_random.integers(4)
        self.current_player = (self.dealer + 1) % 4
        self.trump_suit = None
        self.phase = 0  
        self.tricks = []
        self.played_cards = []
        self.current_trick = []
        self.trick_winner = None
        self.player_cards_taken = [-1,-1,-1,-1]
        self.number_of_rounds_played = 0
        self.playing_iterator = 0
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
            self._handle_choose_suit(action)
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
        if 11 <=action <= 18: #bid 5-double moon, works normally. assumes you pass in a valid bid
            self.current_bid = action - 6
            self.current_high_bidder = self.current_player
            
        self.current_player = (self.current_player + 1) % 4
        if self.current_player == (self.dealer + 1) % 4:
            self.phase = 1  # Move to choosing suit phase

    def _handle_choose_suit(self,action):
        if (self.current_player == self.current_high_bidder):
            self.trump_suit = action-19
            self.phase = 2
        self.current_player = (self.current_player + 1) % 4

    def _handle_play(self, action):
        if (action < 22 ):
            card = self.hands[self.current_player][action]
            self.current_trick.append((card,self.current_player))
            self.hands[self.current_player].remove(card)
            self.played_cards.append(card)
        #TODO fix this
        if self.playing_iterator == 3: #tell when everyone has played:
            self._resolve_trick()
            self.playing_iterator = 0
            self.current_player = (self.current_player + 1) % 4 
            return

        self.current_player = (self.current_player + 1) % 4
        self.playing_iterator += 1


    def _discard_and_fill(self):
        dealerOffset = self.dealer
        for player in range(4):
            self.hands[(player + dealerOffset) % 4] = [card for card in self.hands[(player + dealerOffset) % 4]
                                  if card.suit == self.trump_suit or card.rank == 11]
            while len(self.hands[(player + dealerOffset) % 4]) < 6 and len(self.deck) > 0:
                self.hands[(player + dealerOffset) % 4].append(self.deck.pop())
                self.player_cards_taken[(player + dealerOffset) % 4] += 1
            #bidder team fill phase
            sortLambda = lambda x: 0 if self._is_valid_play(x) else 1 # put the invalid cards at the back of the hand
            bidderInvalidCount = len(filter(lambda card: not self._is_valid_play(card),self.hands[self.current_high_bidder]))
            bidderPartnerInvalidCount = len(filter(lambda card: not self._is_valid_play(card),self.hands[self.current_high_bidder + 2 % 4]))
            self.hands[self.current_high_bidder].sort(key=sortLambda)
            self.hands[self.current_high_bidder + 2 % 4].sort(key=sortLambda)
            while (bidderInvalidCount > 0 and self.deck.count() > 0):
                if (bidderInvalidCount == self.deck.count()):
                    for x in range(0,bidderInvalidCount):
                        self.hands[self.current_high_bidder].insert(0,self.deck.pop())
                else:
                    possibleCard = self.deck.pop()
                    if (self._is_valid_play(possibleCard)):
                        self.hands[self.current_high_bidder].pop()
                        self.hands[self.current_high_bidder].insert(0,possibleCard)
                        bidderInvalidCount = bidderInvalidCount - 1
            if (self.deck.count() == 0):
                return
            else:
                cardsToAddToPartner = filter(lambda card: self._is_valid_play(card),self.deck)
                if (len(cardsToAddToPartner) > bidderPartnerInvalidCount):
                    self.hands[self.current_high_bidder + 2 % 4] = filter(lambda card: self.hands[self.current_high_bidder + 2 % 4])
                    self.hands[self.current_high_bidder + 2 % 4] = self.hands[self.current_high_bidder + 2 % 4] + cardsToAddToPartner
                    self.player_cards_taken[self.current_high_bidder + 2 % 4] = len(cardsToAddToPartner)
                else:
                    for x in range(len(cardsToAddToPartner)):
                        self.hands[self.current_high_bidder + 2 % 4].pop()
                        self.hands[self.current_high_bidder + 2 % 4].insert(0,cardsToAddToPartner.pop())
                    self.player_cards_taken[self.current_high_bidder + 2 % 4] = len(cardsToAddToPartner + 1) #TODO check on this

        
            

    def _resolve_trick(self):
        winning_card_player_tuple = max(self.current_trick, key=lambda c: (self._is_valid_play(c[0]), c[0].rank))
        self.trick_winner = winning_card_player_tuple[1]
        self.tricks.append(self.current_trick)
        self.current_trick = []
        self.current_player = self.trick_winner

        trick_points = sum(self._card_points(tuple[0]) for tuple in self.tricks[-1])
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
        return abs(self.scores[0] - self.scores[1]) > 53 or (self.scores[0] > 53 and self.current_high_bidder % 2 == 0 ) or (self.scores[1] > 53 and self.current_high_bidder % 2 == 1 )

    def _check_current_player_win(self):
        if (self.current_player % 2 == 0):
            return self.scores[0] - self.scores[1] > 53 or (self.scores[0] > 53 and self.current_high_bidder % 2 == 0)
        return self.scores[1] - self.scores[0] > 53 or (self.scores[1] > 53 and self.current_high_bidder % 2 == 1)

    def _calculate_reward(self):
        return (self.scores[self.current_player % 2] - self.scores[(self.current_player + 1) % 2]) - self.number_of_rounds_played*.4 + (2000 if self._check_current_player_win() else 0)  

    def _get_observation(self):
        return {
            'hand': np.array([(card.suit.value if card.suit else 4, card.rank) for card in self.hands[self.current_player]]),
            'played_cards': np.array([(card.suit.value if card.suit else 4, card.rank) for card in self.played_cards]),
            'tricks': np.array([(card.suit.value if card.suit else 4, card.rank) for card in self.current_trick]),
            'scores': np.array(self.scores),
            'current_bid': self.current_bid,
            'current_high_bidder': self.current_high_bidder,
            'dealer': self.dealer,
            'current_player': self.current_player,
            'trump_suit': self.trump_suit if self.trump_suit else 4,
            'phase': self.phase,
            'number_of_rounds_played': self.number_of_rounds_played,
            'player_cards_taken': self.player_cards_taken,
            'action_mask': self._get_action_mask()
        }

    def _get_action_mask(self):
        mask = np.zeros(self.num_actions, dtype=np.int8) #17 = shoot moon
        if self.phase == 0:  # Bidding phase
            currentBidAsMask = self.current_bid + 6
            if (self.current_bid == 0):
                mask[10] = self.current_player != self.dealer
            else:
                mask[10] = 1
            if (currentBidAsMask < 14):
                mask[currentBidAsMask+1 if self.current_bid > 0 else 11:18] = 1 #TODO refactor this
            if (self.current_player == self.dealer):
                if (currentBidAsMask == 14):
                    mask[18] = 1 # double shoot
        elif self.phase == 1: # suit selection phase
            mask[19:23] = 1
        elif self.phase == 2: # Playing phase
            anyTrue = False  
            for i, card in enumerate(self.hands[self.current_player]):
                if self._is_valid_play(card):
                    mask[i] = 1
                    anyTrue = True
            if not anyTrue:
                mask = np.zeros(self.num_actions, dtype=np.int8)
                mask[23] = 1
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

    #returns if a card is a valid play for current game
    def _is_valid_play(self, card):
        return card.suit == self.trump_suit or card.rank == 11 or self._is_off_jack(card) 
