import gymnasium as gym
import numpy as np
import os
from enum import Enum
import json

#TODO LIST
# - fix mask numbers
# figure out how to end the 'playing phase'
# write unit tests for discard and fill
# fix discard and fill
# add observations for how many cards each player takes
# assume burn worst cards-always the right play 

class Suit(Enum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3

class Phase(Enum):
    BIDDING = 0
    CHOOSESUIT = 1
    PLAYING = 2

class Card:
    def __init__(self, suit: Suit, rank: int):
        self.suit = suit
        self.rank = rank  # 2-10 for number cards, 11 for Joker, 12-15 for J,Q,K,A

    def __str__(self):
        ranks = {11: 'Joker', 12: 'J', 13: 'Q', 14: 'K', 15: 'A'}
        return f"{ranks.get(self.rank, self.rank)} of {self.suit.name if self.suit else ''}"

    def __lt__(self, other):
        return self.rank < other.rank
    
    def default(self):
        return {'suit': self.suit, 'rank': self.rank}

class CardEncoder(json.JSONEncoder):
    def default(self,o):
        if (isinstance(o,Card)):
            return o.default()
        if (isinstance(o,Suit)):
            return o.value
        if (isinstance(o,Phase)):
            return o.value
        if (isinstance(o,np.int8) or isinstance(o,np.int64)):
            return int(o)
        return super().default(o)
        


class PitchEnv(gym.Env):
    def __init__(self, win_threshold=54):
        super(PitchEnv, self).__init__()
        self.win_threshold = win_threshold
        self.played_cards = []
        self.num_actions = 24  # 10 for cards in hand, 9 for possible bids (pass, 5-10, shoot the moon, double shoot the moon), 4 for choosing suit, one for no valid play
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Dict({
            'hand': gym.spaces.Box(low=0, high=15, shape=(10, 2), dtype=np.int8), # possible to have up to ten cards
            'tricks': gym.spaces.Box(low=0, high=15, shape=(4, 2), dtype=np.int8),
            'round_scores': gym.spaces.Box(low=0,high=10,shape=(2,), dtype=np.int8), # temporary score used to evaluate if you made your bid or not
            'played_cards': gym.spaces.Box(low=0, high=15, shape=(24, 2), dtype=np.int8),
            'scores': gym.spaces.Box(low=-32768, high=32767, shape=(2,), dtype=np.int16),
            'round_scores': gym.spaces.Box(low=0,high=10,shape=(2,), dtype=np.int8),
            'current_trick': gym.spaces.Box(low=0, high=15, shape=(4, 3), dtype=np.int8),
            'current_bid': gym.spaces.Discrete(9),  # 0,5-10,moon,double moon where 0 means no bid yet
            'current_high_bidder': gym.spaces.Discrete(5), #no bid or 1-4
            'dealer': gym.spaces.Discrete(4),
            'number_of_rounds_played': gym.spaces.Box(low=0,high=2**63 - 2, dtype=np.uint64), # how many rounds has the game gone
            'current_player': gym.spaces.Discrete(4),
            'player_cards_taken': gym.spaces.Box(low=-1,high=10, shape=(4,),dtype=np.int8),
            'trump_suit': gym.spaces.Discrete(5),  # 0-3 for suits, 4 for no trump
            'phase': gym.spaces.Discrete(3),  # 0: bidding, 1: choosing suit, 2: playing
            'action_mask': gym.spaces.Box(low=0, high=1, shape=(self.num_actions,), dtype=np.int8),
            'derived_features': gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
        })
        self.reset()

    def loadGameState(self, deck: Card, hands, round_scores,
                 scores, current_bid, current_high_bidder, dealer, current_player,
                 trump_suit,phase,tricks,played_cards,current_trick,trick_winner,
                 player_cards_taken, number_of_rounds_played, playing_iterator):
        super(PitchEnv, self).__init__()
        self.played_cards = []
        self.num_actions = 24  # 10 for cards in hand, 9 for possible bids (pass, 5-10, shoot the moon, double shoot the moon), 4 for choosing suit, one for no valid play
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Dict({
            'hand': gym.spaces.Box(low=0, high=15, shape=(10, 2), dtype=np.int8), # possible to have up to ten cards
            'tricks': gym.spaces.Box(low=0, high=15, shape=(4, 2), dtype=np.int8),
            'round_scores': gym.spaces.Box(low=0,high=10,shape=(2,), dtype=np.int8), # temporary score used to evaluate if you made your bid or not
            'played_cards': gym.spaces.Box(low=0, high=15, shape=(24, 2), dtype=np.int8),
            'scores': gym.spaces.Box(low=-32768, high=32767, shape=(2,), dtype=np.int16),
            'round_scores': gym.spaces.Box(low=0,high=10,shape=(2,), dtype=np.int8),
            'current_trick': gym.spaces.Box(low=0, high=15, shape=(4, 3), dtype=np.int8),
            'current_bid': gym.spaces.Discrete(9),  # 0,5-10,moon,double moon where 0 means no bid yet
            'current_high_bidder': gym.spaces.Discrete(5), #no bid or 1-4
            'dealer': gym.spaces.Discrete(4),
            'number_of_rounds_played': gym.spaces.Box(low=0,high=2**63 - 2, dtype=np.uint64), # how many rounds has the game gone
            'current_player': gym.spaces.Discrete(4),
            'player_cards_taken': gym.spaces.Box(low=-1,high=10, shape=(4,),dtype=np.int8),
            'trump_suit': gym.spaces.Discrete(5),  # 0-3 for suits, 4 for no trump
            'phase': gym.spaces.Discrete(3),  # 0: bidding, 1: choosing suit, 2: playing
            'action_mask': gym.spaces.Box(low=0, high=1, shape=(self.num_actions,), dtype=np.int8),
            'derived_features': gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
        })
        try:
            self.deck = deck
            self.hands = hands
            self.round_scores = round_scores
            self.scores = scores
            self.current_bid = current_bid
            self.current_high_bidder = current_high_bidder
            self.dealer = dealer
            self.current_player = current_player
            self.trump_suit = trump_suit
            self.phase = phase
            self.tricks = tricks
            self.played_cards = played_cards
            self.current_trick = current_trick
            self.trick_winner = trick_winner
            self.player_cards_taken = player_cards_taken
            self.number_of_rounds_played = number_of_rounds_played
            self.playing_iterator = playing_iterator
        except:
            raise Exception("Unable to load game state from inputted arguments")
        return
    
    # write a constructor that takes in each of the arguments then use **var in json decoder def __init__(self,)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.deck = self._create_deck()
        self.hands: list[list[Card]] = [[] for _ in range(4)]
        self.round_scores: list[int] = [0,0]
        self.scores = [0, 0]
        self.current_bid: int = 0
        self.current_high_bidder: int = 0
        self.dealer: int = self.np_random.integers(0,4,endpoint=False)
        self.current_player: int = (self.dealer + 1) % 4
        self.trump_suit: Suit | None = None
        self.phase: Phase = Phase.BIDDING
        self.tricks: list[list[tuple[Card,int]]] = []
        self.played_cards: list[Card] = []
        self.current_trick: list[tuple[Card,int]] = []
        self.trick_winner = None
        self.player_cards_taken = [-1,-1,-1,-1]
        self.number_of_rounds_played = 0
        self.playing_iterator = 0
        self.last_trick_points = [0, 0]
        self._deal_cards()
        observation = self._get_observation()
        info = {}
        return observation, info

    def deep_copy(self) -> 'PitchEnv':
        """Fast clone for MCTS simulations. Bypasses __init__/reset."""
        clone = object.__new__(PitchEnv)
        # Gymnasium bookkeeping (shared, not mutated)
        clone.num_actions = self.num_actions
        clone.action_space = self.action_space
        clone.observation_space = self.observation_space
        clone.np_random = self.np_random
        clone.win_threshold = self.win_threshold
        # Scalars (immutable)
        clone.current_bid = self.current_bid
        clone.current_high_bidder = self.current_high_bidder
        clone.dealer = self.dealer
        clone.current_player = self.current_player
        clone.trump_suit = self.trump_suit
        clone.phase = self.phase
        clone.trick_winner = self.trick_winner
        clone.number_of_rounds_played = self.number_of_rounds_played
        clone.playing_iterator = self.playing_iterator
        # Lists of primitives
        clone.scores = list(self.scores)
        clone.round_scores = list(self.round_scores)
        clone.player_cards_taken = list(self.player_cards_taken)
        clone.last_trick_points = list(self.last_trick_points)
        # Lists of Cards (Card attrs never mutated)
        clone.deck = list(self.deck)
        clone.played_cards = list(self.played_cards)
        clone.hands = [list(h) for h in self.hands]
        # Nested lists of tuples (tuples are immutable)
        clone.current_trick = list(self.current_trick)
        clone.tricks = [list(t) for t in self.tricks]
        return clone

    def print_state(self): #sometimes you just gotta debug
        strn = '\nCurrent State: \n'
        strn += 'Current Phase: ' + str(self.phase) + '\n'
        strn += 'Current Bid: ' + str(self.current_bid) + '\n'
        strn += 'Current Bidder: ' + str(self.current_high_bidder) + '\n'
        strn += 'Dealer: ' + str(self.dealer) + '\n'
        strn += 'Current Player: ' + str(self.current_player) + '\n'
        strn += f'Player 0 hand: {list(map(lambda card: (card.suit,card.rank),self.hands[0]))}\n'
        strn += f'Player 1 hand: {list(map(lambda card: (card.suit,card.rank),self.hands[1]))}\n'
        strn += f'Player 2 hand: {list(map(lambda card: (card.suit,card.rank),self.hands[2]))}\n'
        strn += f'Player 3 hand: {list(map(lambda card: (card.suit,card.rank),self.hands[3]))}\n'
        strn += f'Trump suit: {self.trump_suit}\n'
        strn += f'Current Trick: {list(map(lambda tuple: (tuple[0].suit,tuple[0].rank,tuple[1]),self.current_trick))}\n'
        strn += f'Current round scores: {self.round_scores}\n'
        strn += f'Current action mask: {self._get_action_mask()}'
        print(strn)
        return
    
    def saveStateToFileAsJson(self,fileName: str):
        tmpExtension: int = 0 
        filepath = './' + fileName + str(tmpExtension) + '.json'
        while (os.path.exists(filepath)):
            tmpExtension += 1
            filepath= './' + fileName + str(tmpExtension) + '.json'

        outPutState: dict = {'deck': self.deck,
                       'hands': self.hands,
                       'round_scores': self.round_scores,
                       'scores': self.scores,
                       'current_bid': self.current_bid,
                       'current_high_bidder': self.current_high_bidder,
                       'dealer': self.dealer,
                       'current_player': self.current_player,
                       'trump_suit': self.trump_suit,
                       'phase': self.phase,
                       'tricks': self.tricks,
                       'played_cards': self.played_cards,
                       'current_trick': self.current_trick,
                       'trick_winner': self.trick_winner,
                       'player_cards_taken': self.player_cards_taken,
                       'number_of_rounds_played': self.number_of_rounds_played,
                       'playing_iterator': self.playing_iterator}
        outStr: str = CardEncoder().encode(outPutState)
        f = open('./' + fileName + str(tmpExtension) + '.json','w')
        f.write(outStr)
        f.close()

    def loadStateFromJsonString(self,jsonStr: str):
        parsedObj = json.loads(jsonStr)
        
        lambdaToCard = lambda card: Card(card['suit'],card['rank'])

        deck = map(lambdaToCard,parsedObj['deck'])
        
        hands = map(lambda arr: map(lambdaToCard,arr),parsedObj['hands'])

        roundScores = parsedObj['round_scores']

        scores = parsedObj['scores']

        tricks = map(lambda listOfListOfTuples: map(lambda listOfTuples: map(lambda tuple: (lambdaToCard(tuple[0]),tuple[1]),listOfTuples), listOfListOfTuples),parsedObj['tricks'])

        playedCards = map(lambdaToCard,parsedObj['played_cards'])

        currentTrick = map(lambda listOfTuples: map(lambda tuple: (lambdaToCard(tuple[0]),tuple[1]),listOfTuples),parsedObj['current_trick'])

        self.loadGameState(deck,hands,roundScores,scores,
                            parsedObj['current_bid'],parsedObj['current_high_bidder'],parsedObj['dealer'],parsedObj['current_player'],parsedObj['trump_suit']
                            ,parsedObj['phase'],tricks,playedCards,currentTrick,parsedObj['trick_winner'],parsedObj['player_cards_taken'],parsedObj['number_of_rounds_played'],parsedObj['playing_iterator'])

        pass
        
    def step(self, action, current_obs=None):
        mask = self._get_action_mask()
        if mask[action] == 0:
            valid_actions = np.where(mask == 1)[0]
            action = self.np_random.choice(valid_actions)

        team = self.current_player % 2
        scores_before = list(self.scores)
        self.last_trick_points = [0, 0]

        if self.phase == Phase.BIDDING:  # Bidding phase
            self._handle_bid(action)
        elif self.phase == Phase.CHOOSESUIT:
            self._handle_choose_suit(action)
        elif self.phase == Phase.PLAYING:  # Playing phase
            self._handle_play(action)

        terminated = self._check_game_end()
        truncated = False
        reward = self._calculate_reward(team, scores_before)
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
        elif (action != 10):
            raise Exception(f"invalid action passed to handle bid. Action: {action}")
        
        if self.current_player == self.dealer:
            # Dealer was the last to bid — bidding round complete
            self.current_player = self.current_high_bidder
            self.phase = Phase.CHOOSESUIT  # Move to choosing suit phase
        else:
            self.current_player = (self.current_player + 1) % 4

    def _handle_choose_suit(self,action):
        if (self.current_player == self.current_high_bidder):
            if (not (19 <= action < 23)):
                raise Exception(f"invalid bid action from winning bidder. action: {action}")
            self.trump_suit = Suit(action-19)
            self._discard_and_fill()
            self.phase = Phase.PLAYING
            return
        self.current_player = (self.current_player + 1) % 4

    def _no_more_valid_plays_any_hand(self):
        for i in range(0,4):
            for j in range(0,len(self.hands[i])):
                if (self._is_valid_play(self.hands[i][j])):
                    return False
        return True
        

    def _handle_play(self, action):
        if (action <= 9):
            card = self.hands[self.current_player][action]
            self.current_trick.append((card,self.current_player))
            self.hands[self.current_player].remove(card)
            self.played_cards.append(card)
        if (len(self.current_trick) == 0 and self._no_more_valid_plays_any_hand()):
            self._end_round()
            return
        if self.playing_iterator == 3: #tell when everyone has played:
            if not self.current_trick:
                # All 4 players passed with no valid plays this trick
                self._end_round()
                return
            self._resolve_trick()
            self.playing_iterator = 0
            # Trick winner leads next (set by _resolve_trick);
            # if they have no valid plays, advance clockwise
            if not any(self._is_valid_play(c) for c in self.hands[self.current_player]):
                self.current_player = (self.current_player + 1) % 4
                self.playing_iterator = 1
            return

        self.current_player = (self.current_player + 1) % 4
        self.playing_iterator += 1


    def _discard_and_fill(self):
        bidder = self.current_high_bidder
        partner = (bidder + 2) % 4

        # Phase 1: All players discard non-playable cards (keep trump, jokers, off-jack)
        for player in range(4):
            p = (player + self.dealer) % 4
            self.hands[p] = [card for card in self.hands[p] if self._is_valid_play(card)]

        # Phase 2: All players fill to 6 from deck in dealer order
        for player in range(4):
            p = (player + self.dealer) % 4
            while len(self.hands[p]) < 6 and len(self.deck) > 0:
                self.hands[p].append(self.deck.pop())
                self.player_cards_taken[p] += 1

        # Phase 3: Bidder swaps invalid drawn cards for playable ones from deck,
        # then partner gets the same treatment with whatever is left
        for p in [bidder, partner]:
            invalid_cards = [c for c in self.hands[p] if not self._is_valid_play(c)]
            for bad_card in invalid_cards:
                replacement = None
                for card in self.deck:
                    if self._is_valid_play(card):
                        replacement = card
                        break
                if replacement is not None:
                    self.hands[p].remove(bad_card)
                    self.hands[p].append(replacement)
                    self.deck.remove(replacement)

    def _end_round(self):
        #add the scores up
        if self.current_bid <= 10:
            if (self.round_scores[(self.current_high_bidder) % 2] >= self.current_bid):
                self.scores[self.current_high_bidder % 2] += self.round_scores[self.current_high_bidder % 2]
            else:
                self.scores[self.current_high_bidder % 2] -= self.current_bid
        else: #moon conditions
            if (self.round_scores[(self.current_high_bidder) % 2] == 10):
                self.scores[self.current_high_bidder % 2] += 20 if self.current_bid == 11 else 40
                pass
            else:
                self.scores[self.current_high_bidder % 2] -= 20 if self.current_bid == 11 else 40
        self.scores[(self.current_high_bidder + 1) % 2] += self.round_scores[(self.current_high_bidder + 1) % 2]
        self.dealer = (self.dealer + 1) % 4
        self.deck = self._create_deck()
        self.hands = [[] for _ in range(4)]
        self.round_scores = [0,0]
        self.current_bid = 0
        self.current_player = (self.dealer + 1) % 4
        self.trump_suit = None
        self.phase = Phase.BIDDING
        self.tricks = []
        self.played_cards = []
        self.current_trick = []
        self.trick_winner = None
        self.player_cards_taken = [-1,-1,-1,-1]
        self.number_of_rounds_played += 1
        self.playing_iterator = 0
        self._deal_cards()
        

    def _resolve_trick(self):
        winning_card_player_tuple = max(self.current_trick, key=lambda c: (self._is_valid_play(c[0]), c[0].rank,
                                                                            1 if (c[0].rank == 12 and c[0].suit == self.trump_suit) else 0))
        self.trick_winner = winning_card_player_tuple[1]
        self.tricks.append(self.current_trick)
        self.current_trick = []
        self.current_player = self.trick_winner
        winning_team = self.trick_winner % 2
        self.last_trick_points = [0, 0]
        for card, player in self.tricks[-1]:
            pts = self._card_points(card)
            if pts > 0 and card.rank == 2:
                # 2 of trump scores for the team that played it
                self.round_scores[player % 2] += pts
                self.last_trick_points[player % 2] += pts
            else:
                self.round_scores[winning_team] += pts
                self.last_trick_points[winning_team] += pts

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
        t = self.win_threshold
        if self.number_of_rounds_played >= 50:
            return True
        return abs(self.scores[0] - self.scores[1]) >= t or (self.scores[0] >= t and self.current_high_bidder % 2 == 0 ) or (self.scores[1] >= t and self.current_high_bidder % 2 == 1 )

    def _check_current_player_win(self):
        t = self.win_threshold
        if (self.current_player % 2 == 0):
            return self.scores[0] - self.scores[1] >= t or (self.scores[0] >= t and self.current_high_bidder % 2 == 0)
        return self.scores[1] - self.scores[0] >= t or (self.scores[1] >= t and self.current_high_bidder % 2 == 1)

    def _team_won(self, team):
        """Check if the given team (0 or 1) has won the game."""
        t = self.win_threshold
        other = 1 - team
        return (self.scores[team] - self.scores[other] >= t or
                (self.scores[team] >= t and self.current_high_bidder % 2 == team))

    def _calculate_reward(self, team, scores_before):
        other_team = 1 - team
        # Trick-level: points my team won minus points other team won
        reward = self.last_trick_points[team] - self.last_trick_points[other_team]
        # Round-end: actual score delta captures set penalties, moon bonuses, etc.
        # e.g. bid 7 and only took 4 → scores drop by 7, reward reflects that
        score_delta = (self.scores[team] - scores_before[team]) - (self.scores[other_team] - scores_before[other_team])
        reward += score_delta
        # Game-end bonus: use team (captured before action) not current_player
        # (which may have changed during _end_round)
        if self._check_game_end():
            margin = self.scores[team] - self.scores[other_team]
            if self._team_won(team):
                reward += 10 + min(abs(margin) / 5.0, 10)
            else:
                reward -= 10 + min(abs(margin) / 5.0, 10)
        return reward

    def _get_observation(self):
        # Pad hand to fixed size (10, 2)
        hand_data = [(card.suit.value if card.suit else 4, card.rank) for card in self.hands[self.current_player]]
        hand_padded = np.zeros((10, 2), dtype=np.int8)
        if hand_data:
            hand_padded[:len(hand_data)] = np.array(hand_data, dtype=np.int8)

        # Pad played_cards to fixed size (24, 2)
        played_data = [(card.suit.value if card.suit else 4, card.rank) for card in self.played_cards]
        played_padded = np.zeros((24, 2), dtype=np.int8)
        if played_data:
            played_padded[:len(played_data)] = np.array(played_data, dtype=np.int8)

        # Pad current_trick to fixed size (4, 3)
        trick_data = [(t[0].suit.value if t[0].suit else 4, t[0].rank, t[1]) for t in self.current_trick]
        trick_padded = np.zeros((4, 3), dtype=np.int8)
        if trick_data:
            trick_padded[:len(trick_data)] = np.array(trick_data, dtype=np.int8)

        return {
            'hand': hand_padded,
            'played_cards': played_padded,
            'scores': np.array(self.scores, dtype=np.int16),
            'round_scores': np.array(self.round_scores, dtype=np.int16),
            'current_trick': trick_padded,
            'current_bid': self.current_bid,
            'current_high_bidder': self.current_high_bidder,
            'dealer': self.dealer,
            'current_player': self.current_player,
            'trump_suit': self.trump_suit.value if self.trump_suit is not None else 4,
            'phase': self.phase.value if isinstance(self.phase, Phase) else self.phase,
            'number_of_rounds_played': self.number_of_rounds_played,
            'player_cards_taken': np.array(self.player_cards_taken, dtype=np.int8),
            'action_mask': self._get_action_mask(),
            'derived_features': self._get_derived_features(),
        }

    def _get_derived_features(self) -> np.ndarray:
        """Compute 10 strategic features, all normalized to [0, 1]."""
        feats = np.zeros(10, dtype=np.float32)
        cp = self.current_player
        my_team = cp % 2
        is_playing = (self.phase == Phase.PLAYING)

        if is_playing:
            hand = self.hands[cp]
            trump_cards = [c for c in hand if self._is_valid_play(c)]
            trump_count = len(trump_cards)

            # 0: trump_card_count
            feats[0] = trump_count / 10.0

            # 1: trump_point_count
            trump_pts = sum(self._card_points(c) for c in trump_cards)
            feats[1] = trump_pts / 7.0

            # 2: void_in_trump
            feats[2] = 1.0 if trump_count == 0 else 0.0

            # 3: highest_trump_rank (normalized rank 2-15 → 0-1)
            if trump_cards:
                max_rank = max(c.rank for c in trump_cards)
                feats[3] = (max_rank - 2) / 13.0
            else:
                feats[3] = 0.0

            # Find current trick winner (shared by feats 4 and 5)
            if not self.current_trick:
                feats[4] = 1.0  # leading
                feats[5] = 0.0
            else:
                winner_card, winner_player = max(
                    self.current_trick,
                    key=lambda t: (self._is_valid_play(t[0]), t[0].rank,
                                   1 if (t[0].rank == 12 and t[0].suit == self.trump_suit) else 0))
                # 4: can_win_trick
                feats[4] = 1.0 if any(
                    c.rank > winner_card.rank or
                    (c.rank == 12 and c.suit == self.trump_suit and self._is_off_jack(winner_card))
                    for c in trump_cards
                ) else 0.0
                # 5: partner_winning
                feats[5] = 1.0 if winner_player % 2 == my_team else 0.0

        # 6: am_high_bidder (always)
        feats[6] = 1.0 if self.current_high_bidder == cp else 0.0

        # 7: bid_deficit (always) — max bid is 12 (double moon)
        feats[7] = max(0.0, self.current_bid - self.round_scores[my_team]) / 12.0

        # 8: tricks_remaining (always) — max hand size is 9 at deal, 6 after fill
        max_hand = max((len(h) for h in self.hands), default=0)
        feats[8] = max_hand / 9.0

        # 9: point_cards_remaining (always)
        total_scored = sum(self.round_scores)
        feats[9] = 1.0 - min(total_scored, 9) / 9.0

        return feats

    def _get_action_mask(self):
        mask = np.zeros(self.num_actions, dtype=np.int8) #17 = shoot moon
        if self.phase == Phase.BIDDING:  # Bidding phase
            currentBidAsMask = self.current_bid + 6
            if (self.current_bid == 0):
                mask[10] = self.current_player != self.dealer
            else:
                mask[10] = 1
            if (currentBidAsMask < 17):
                mask[currentBidAsMask+1 if self.current_bid > 0 else 11:18] = 1
            if (self.current_player == self.dealer and currentBidAsMask == 17):
                mask[18] = 1 # dealer can double moon when someone bid moon
        elif self.phase == Phase.CHOOSESUIT: # suit selection phase
            mask[19:23] = 1
        elif self.phase == Phase.PLAYING: # Playing phase
            anyTrue = False  
            for i, card in enumerate(self.hands[self.current_player]):
                if self._is_valid_play(card):
                    mask[i] = 1
                    anyTrue = True
            if not anyTrue:
                mask = np.zeros(self.num_actions, dtype=np.int8)
                mask[23] = 1
        if np.count_nonzero(mask) == 0:
            print('something failed')
            raise Exception
        
        return mask
    
    _OFF_JACK_PAIR = {
        Suit.CLUBS: Suit.SPADES,
        Suit.SPADES: Suit.CLUBS,
        Suit.DIAMONDS: Suit.HEARTS,
        Suit.HEARTS: Suit.DIAMONDS,
    }

    def _is_off_jack(self, card):
        if card.rank != 12:
            return False
        return card.suit == self._OFF_JACK_PAIR.get(self.trump_suit)

    #returns if a card is a valid play for current game
    def _is_valid_play(self, card):
        return card.suit == self.trump_suit or card.rank == 11 or self._is_off_jack(card) 
