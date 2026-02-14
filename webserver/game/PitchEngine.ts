import {
  Suit,
  Phase,
  Card,
  CardData,
  cardToData,
  cardPoints,
  OFF_JACK_MAP,
  ACTION_PASS,
  ACTION_BID_START,
  ACTION_SUIT_START,
  ACTION_NO_VALID_PLAY,
  NUM_ACTIONS,
} from "./constants.js";

function shuffleArray<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export interface TrickEntry {
  card: Card;
  seatIndex: number;
}

export interface RoundEndResult {
  bidderTeam: number;
  bidAmount: number;
  bidMade: boolean;
  roundScores: [number, number];
  scoreDeltas: [number, number];
  totalScores: [number, number];
  newDealer: number;
}

export interface TrickResult {
  trick: TrickEntry[];
  winner: number;
  pointsWon: number;
  roundScores: [number, number];
}

export class PitchEngine {
  winThreshold: number;
  deck: Card[] = [];
  hands: Card[][] = [[], [], [], []];
  roundScores: [number, number] = [0, 0];
  scores: [number, number] = [0, 0];
  currentBid = 0;
  currentHighBidder = 0;
  dealer = 0;
  currentPlayer = 0;
  trumpSuit: Suit | null = null;
  phase: Phase = Phase.BIDDING;
  tricks: TrickEntry[][] = [];
  playedCards: Card[] = [];
  currentTrick: TrickEntry[] = [];
  trickWinner: number | null = null;
  playerCardsTaken: number[] = [-1, -1, -1, -1];
  numberOfRoundsPlayed = 0;
  playingIterator = 0;
  gameOver = false;
  winner: 0 | 1 | null = null;

  constructor(winThreshold = 54) {
    this.winThreshold = winThreshold;
  }

  reset(dealerOverride?: number): void {
    this.deck = this.createDeck();
    this.hands = [[], [], [], []];
    this.roundScores = [0, 0];
    this.scores = [0, 0];
    this.currentBid = 0;
    this.currentHighBidder = 0;
    this.dealer = dealerOverride ?? Math.floor(Math.random() * 4);
    this.currentPlayer = (this.dealer + 1) % 4;
    this.trumpSuit = null;
    this.phase = Phase.BIDDING;
    this.tricks = [];
    this.playedCards = [];
    this.currentTrick = [];
    this.trickWinner = null;
    this.playerCardsTaken = [-1, -1, -1, -1];
    this.numberOfRoundsPlayed = 0;
    this.playingIterator = 0;
    this.gameOver = false;
    this.winner = null;
    this.dealCards();
  }

  /** Start a new round (preserves scores, increments dealer) */
  newRound(): void {
    this.dealer = (this.dealer + 1) % 4;
    this.deck = this.createDeck();
    this.hands = [[], [], [], []];
    this.roundScores = [0, 0];
    this.currentBid = 0;
    this.currentHighBidder = 0;
    this.currentPlayer = (this.dealer + 1) % 4;
    this.trumpSuit = null;
    this.phase = Phase.BIDDING;
    this.tricks = [];
    this.playedCards = [];
    this.currentTrick = [];
    this.trickWinner = null;
    this.playerCardsTaken = [-1, -1, -1, -1];
    this.numberOfRoundsPlayed++;
    this.playingIterator = 0;
    this.dealCards();
  }

  getActionMask(): number[] {
    const mask = new Array(NUM_ACTIONS).fill(0);

    if (this.phase === Phase.BIDDING) {
      const currentBidAsMask = this.currentBid + 6;
      if (this.currentBid === 0) {
        mask[ACTION_PASS] = this.currentPlayer !== this.dealer ? 1 : 0;
      } else {
        mask[ACTION_PASS] = 1;
      }
      if (currentBidAsMask < 14) {
        const start = this.currentBid > 0 ? currentBidAsMask + 1 : 11;
        for (let i = start; i < 18; i++) {
          mask[i] = 1;
        }
      }
      if (this.currentPlayer === this.dealer) {
        if (currentBidAsMask === 14) {
          // current bid is 8 (8+6=14), dealer can double
          mask[18] = 1;
        }
      }
    } else if (this.phase === Phase.CHOOSESUIT) {
      if (this.currentPlayer === this.currentHighBidder) {
        for (let i = 19; i < 23; i++) {
          mask[i] = 1;
        }
      }
    } else if (this.phase === Phase.PLAYING) {
      let anyTrue = false;
      for (let i = 0; i < this.hands[this.currentPlayer].length; i++) {
        if (this.isValidPlay(this.hands[this.currentPlayer][i])) {
          mask[i] = 1;
          anyTrue = true;
        }
      }
      if (!anyTrue) {
        mask.fill(0);
        mask[ACTION_NO_VALID_PLAY] = 1;
      }
    }

    return mask;
  }

  /**
   * Execute an action. Returns metadata about what happened for broadcasting.
   */
  step(action: number): {
    type: "bid" | "trumpChosen" | "cardPlayed" | "noValidPlay" | "trickResult" | "roundEnd" | "gameOver";
    data: Record<string, unknown>;
  }[] {
    const events: { type: string; data: Record<string, unknown> }[] = [];
    const mask = this.getActionMask();

    // If invalid action, pick random valid one (safety fallback)
    if (mask[action] !== 1) {
      const valid = mask.map((v, i) => (v === 1 ? i : -1)).filter((i) => i >= 0);
      action = valid[Math.floor(Math.random() * valid.length)];
    }

    if (this.phase === Phase.BIDDING) {
      events.push(...this.handleBid(action));
    } else if (this.phase === Phase.CHOOSESUIT) {
      events.push(...this.handleChooseSuit(action));
    } else if (this.phase === Phase.PLAYING) {
      events.push(...this.handlePlay(action));
    }

    return events as { type: "bid" | "trumpChosen" | "cardPlayed" | "noValidPlay" | "trickResult" | "roundEnd" | "gameOver"; data: Record<string, unknown> }[];
  }

  getHandData(seatIndex: number): CardData[] {
    return this.hands[seatIndex].map(cardToData);
  }

  getPlayerCardCounts(): [number, number, number, number] {
    return [
      this.hands[0].length,
      this.hands[1].length,
      this.hands[2].length,
      this.hands[3].length,
    ];
  }

  // --- Private methods ---

  private createDeck(): Card[] {
    const deck: Card[] = [];
    for (const suit of [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]) {
      for (let rank = 2; rank <= 10; rank++) {
        deck.push({ suit, rank });
      }
      for (let rank = 12; rank <= 15; rank++) {
        deck.push({ suit, rank });
      }
    }
    // Two jokers
    deck.push({ suit: null, rank: 11 });
    deck.push({ suit: null, rank: 11 });
    return shuffleArray(deck);
  }

  private dealCards(): void {
    for (let i = 0; i < 9; i++) {
      for (let p = 0; p < 4; p++) {
        this.hands[p].push(this.deck.pop()!);
      }
    }
  }

  private handleBid(action: number): { type: string; data: Record<string, unknown> }[] {
    const events: { type: string; data: Record<string, unknown> }[] = [];
    const actingSeat = this.currentPlayer;

    if (action >= 11 && action <= 18) {
      this.currentBid = action - 6;
      this.currentHighBidder = this.currentPlayer;
      events.push({
        type: "bid",
        data: { seatIndex: actingSeat, action: this.currentBid },
      });
    } else if (action === ACTION_PASS) {
      events.push({
        type: "bid",
        data: { seatIndex: actingSeat, action: "pass" },
      });
    }

    if (actingSeat === this.dealer) {
      // Dealer was the last to bid — bidding round complete
      this.currentPlayer = this.currentHighBidder;
      this.phase = Phase.CHOOSESUIT;
    } else {
      this.currentPlayer = (this.currentPlayer + 1) % 4;
    }

    return events;
  }

  private handleChooseSuit(action: number): { type: string; data: Record<string, unknown> }[] {
    const events: { type: string; data: Record<string, unknown> }[] = [];

    if (this.currentPlayer === this.currentHighBidder) {
      if (action >= 19 && action < 23) {
        this.trumpSuit = action - 19;
        this.discardAndFill();
        this.phase = Phase.PLAYING;
        events.push({
          type: "trumpChosen",
          data: {
            suit: this.trumpSuit,
            bidder: this.currentHighBidder,
            bidAmount: this.currentBid,
          },
        });
      }
    } else {
      this.currentPlayer = (this.currentPlayer + 1) % 4;
    }

    return events;
  }

  private handlePlay(action: number): { type: string; data: Record<string, unknown> }[] {
    const events: { type: string; data: Record<string, unknown> }[] = [];

    if (action <= 9) {
      const card = this.hands[this.currentPlayer][action];
      this.currentTrick.push({ card, seatIndex: this.currentPlayer });
      this.hands[this.currentPlayer].splice(action, 1);
      this.playedCards.push(card);
      events.push({
        type: "cardPlayed",
        data: {
          seatIndex: this.currentPlayer,
          card: cardToData(card),
          handIndex: action,
        },
      });
    } else if (action === ACTION_NO_VALID_PLAY) {
      events.push({
        type: "noValidPlay",
        data: { seatIndex: this.currentPlayer },
      });
    }

    // Check if trick is empty and no valid plays anywhere (round end)
    if (this.currentTrick.length === 0 && this.noMoreValidPlaysAnyHand()) {
      events.push(...this.endRound());
      return events;
    }

    if (this.playingIterator === 3) {
      // All 4 players have acted this trick
      const trickResult = this.resolveTrick();
      events.push({ type: "trickResult", data: trickResult as unknown as Record<string, unknown> });
      this.playingIterator = 0;
      // Trick winner leads next; if they have no valid plays, advance clockwise
      if (!this.playerHasValidPlay(this.currentPlayer)) {
        this.currentPlayer = (this.currentPlayer + 1) % 4;
      }
      return events;
    }

    this.currentPlayer = (this.currentPlayer + 1) % 4;
    this.playingIterator++;

    return events;
  }

  private discardAndFill(): void {
    const bidder = this.currentHighBidder;
    const partner = (bidder + 2) % 4;

    // Phase 1: All players discard non-playable cards
    for (let player = 0; player < 4; player++) {
      const p = (player + this.dealer) % 4;
      this.hands[p] = this.hands[p].filter((card) => this.isValidPlay(card));
    }

    // Phase 2: All players fill to 6 from deck in dealer order
    for (let player = 0; player < 4; player++) {
      const p = (player + this.dealer) % 4;
      while (this.hands[p].length < 6 && this.deck.length > 0) {
        this.hands[p].push(this.deck.pop()!);
        this.playerCardsTaken[p]++;
      }
    }

    // Phase 3: Bidder swaps invalid drawn cards, then partner
    for (const p of [bidder, partner]) {
      const invalidCards = this.hands[p].filter((c) => !this.isValidPlay(c));
      for (const badCard of invalidCards) {
        let replacement: Card | null = null;
        for (const card of this.deck) {
          if (this.isValidPlay(card)) {
            replacement = card;
            break;
          }
        }
        if (replacement !== null) {
          const idx = this.hands[p].indexOf(badCard);
          this.hands[p].splice(idx, 1);
          this.hands[p].push(replacement);
          const deckIdx = this.deck.indexOf(replacement);
          this.deck.splice(deckIdx, 1);
        }
      }
    }
  }

  private playerHasValidPlay(seat: number): boolean {
    return this.hands[seat].some((card) => this.isValidPlay(card));
  }

  private noMoreValidPlaysAnyHand(): boolean {
    for (let i = 0; i < 4; i++) {
      if (this.playerHasValidPlay(i)) return false;
    }
    return true;
  }

  private endRound(): { type: string; data: Record<string, unknown> }[] {
    const events: { type: string; data: Record<string, unknown> }[] = [];
    const scoresBefore: [number, number] = [...this.scores];

    // Score the round
    if (this.currentBid <= 10) {
      if (this.roundScores[this.currentHighBidder % 2] >= this.currentBid) {
        this.scores[this.currentHighBidder % 2] += this.roundScores[this.currentHighBidder % 2];
      } else {
        this.scores[this.currentHighBidder % 2] -= this.currentBid;
      }
    } else {
      // Moon conditions
      if (this.roundScores[this.currentHighBidder % 2] === 10) {
        this.scores[this.currentHighBidder % 2] += this.currentBid === 11 ? 20 : 40;
      } else {
        this.scores[this.currentHighBidder % 2] -= this.currentBid === 11 ? 20 : 40;
      }
    }

    // Non-bidding team always banks their points
    this.scores[(this.currentHighBidder + 1) % 2] += this.roundScores[(this.currentHighBidder + 1) % 2];

    const scoreDeltas: [number, number] = [
      this.scores[0] - scoresBefore[0],
      this.scores[1] - scoresBefore[1],
    ];

    const bidMade = this.currentBid <= 10
      ? this.roundScores[this.currentHighBidder % 2] >= this.currentBid
      : this.roundScores[this.currentHighBidder % 2] === 10;

    const roundEndData: RoundEndResult = {
      bidderTeam: this.currentHighBidder % 2,
      bidAmount: this.currentBid,
      bidMade,
      roundScores: [...this.roundScores],
      scoreDeltas,
      totalScores: [...this.scores],
      newDealer: (this.dealer + 1) % 4,
    };

    events.push({ type: "roundEnd", data: roundEndData as unknown as Record<string, unknown> });

    // Check game end
    if (this.checkGameEnd()) {
      this.gameOver = true;
      this.winner = this.scores[0] >= this.scores[1] ? 0 : 1;
      events.push({
        type: "gameOver",
        data: {
          winner: this.winner,
          finalScores: [...this.scores],
          reason: this.numberOfRoundsPlayed >= 50 ? "maxRounds" : "threshold",
        },
      });
    } else {
      // Start a new round
      this.newRound();
    }

    return events;
  }

  private resolveTrick(): TrickResult {
    // Find winning card — highest rank among valid plays
    let winnerIdx = 0;
    for (let i = 1; i < this.currentTrick.length; i++) {
      const challenger = this.currentTrick[i];
      const current = this.currentTrick[winnerIdx];
      const challengerValid = this.isValidPlay(challenger.card);
      const currentValid = this.isValidPlay(current.card);
      if (challengerValid && !currentValid) {
        winnerIdx = i;
      } else if (challengerValid && currentValid && challenger.card.rank > current.card.rank) {
        winnerIdx = i;
      }
    }

    const winnerEntry = this.currentTrick[winnerIdx];
    this.trickWinner = winnerEntry.seatIndex;
    this.tricks.push([...this.currentTrick]);

    const winningTeam = this.trickWinner % 2;
    let pointsWon = 0;
    for (const entry of this.currentTrick) {
      const pts = cardPoints(entry.card);
      if (pts > 0 && entry.card.rank === 2) {
        // 2 of trump scores for the team that played it
        this.roundScores[entry.seatIndex % 2] += pts;
      } else {
        pointsWon += pts;
      }
    }
    this.roundScores[winningTeam] += pointsWon;
    pointsWon += this.currentTrick.reduce((sum, e) => sum + (e.card.rank === 2 ? cardPoints(e.card) : 0), 0);

    const result: TrickResult = {
      trick: this.currentTrick.map((e) => ({ card: e.card, seatIndex: e.seatIndex })),
      winner: this.trickWinner,
      pointsWon,
      roundScores: [...this.roundScores],
    };

    this.currentTrick = [];
    this.currentPlayer = this.trickWinner;

    return result;
  }

  private checkGameEnd(): boolean {
    const t = this.winThreshold;
    if (this.numberOfRoundsPlayed >= 50) return true;
    return (
      Math.abs(this.scores[0] - this.scores[1]) >= t ||
      (this.scores[0] >= t && this.currentHighBidder % 2 === 0) ||
      (this.scores[1] >= t && this.currentHighBidder % 2 === 1)
    );
  }

  isOffJack(card: Card): boolean {
    if (card.rank !== 12) return false;
    if (this.trumpSuit === null || card.suit === null) return false;
    return OFF_JACK_MAP[this.trumpSuit] === card.suit;
  }

  isValidPlay(card: Card): boolean {
    return card.suit === this.trumpSuit || card.rank === 11 || this.isOffJack(card);
  }
}
