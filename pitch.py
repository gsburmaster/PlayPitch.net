from deck_of_cards import deck_of_cards as deck

cards = deck.DeckOfCards()
cards.add_jokers()

class Player:
    def __init__(self,id) -> None:
        self.id = id
    def setHand(self,hand):
        self.hand = hand
    def setHuman(self,bool):
        self.Human = bool
    
class Game:
    def __init__(self) -> None:
        self.cards = deck.DeckOfCards()
        self.cards.add_jokers()
        self.score = [0,0]
        self.players = [Player(0),Player(1),Player(2),Player(3)]
        self.bidder = self.players[0].id
        self.dealer = self.players[0].id
        self.suit = 0
        self.bids = [-1,-1,-1,-1]
    def playAs(self,id):
        self.players[id].setHuman = True
    def checkWin(self): #returns 0 if no win, 1 if team 1, 2 if team 2
        if (self.score[0] < 54 and self.score[1] < 54):
            return 0
        if (self.score[0] > 54 and self.score[1] < 54 and self.bidder.id in (0,2)):
            return 1
        if (self.score[0] < 54 and self.score[1] > 54 and self.bidder.id in (1,3)):
            return 2
        
    def biddingRound(self):
        True