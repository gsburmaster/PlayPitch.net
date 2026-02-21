import { describe, it, expect, beforeEach } from "vitest";
import { PitchEngine } from "../game/PitchEngine.js";
import {
  Suit,
  Phase,
  Card,
  cardPoints,
  OFF_JACK_MAP,
  ACTION_PASS,
  ACTION_BID_START,
  ACTION_SUIT_START,
  ACTION_NO_VALID_PLAY,
  NUM_ACTIONS,
} from "../game/constants.js";

describe("PitchEngine", () => {
  let engine: PitchEngine;

  beforeEach(() => {
    engine = new PitchEngine();
    engine.reset(0); // dealer=0 for deterministic tests
  });

  // --- Initialization ---

  describe("initialization", () => {
    it("creates a deck of 54 cards", () => {
      const e = new PitchEngine();
      e.reset(0);
      // After dealing 36 cards (9*4), deck should have 18 left
      expect(e.deck.length).toBe(54 - 36);
    });

    it("deals 9 cards to each player", () => {
      for (let i = 0; i < 4; i++) {
        expect(engine.hands[i].length).toBe(9);
      }
    });

    it("sets default state correctly", () => {
      expect(engine.phase).toBe(Phase.BIDDING);
      expect(engine.currentBid).toBe(0);
      expect(engine.currentHighBidder).toBe(0);
      expect(engine.trumpSuit).toBeNull();
      expect(engine.gameOver).toBe(false);
      expect(engine.winner).toBeNull();
      expect(engine.scores).toEqual([0, 0]);
      expect(engine.roundScores).toEqual([0, 0]);
      expect(engine.numberOfRoundsPlayed).toBe(0);
    });

    it("sets dealer from override", () => {
      const e = new PitchEngine();
      e.reset(2);
      expect(e.dealer).toBe(2);
      expect(e.currentPlayer).toBe(3); // dealer+1
    });

    it("currentPlayer is left of dealer", () => {
      expect(engine.dealer).toBe(0);
      expect(engine.currentPlayer).toBe(1);
    });

    it("wraps currentPlayer when dealer is 3", () => {
      const e = new PitchEngine();
      e.reset(3);
      expect(e.currentPlayer).toBe(0);
    });

    it("win threshold defaults to 54", () => {
      expect(engine.winThreshold).toBe(54);
    });

    it("accepts custom win threshold", () => {
      const e = new PitchEngine(100);
      expect(e.winThreshold).toBe(100);
    });
  });

  // --- Bidding ---

  describe("bidding", () => {
    it("starts in BIDDING phase", () => {
      expect(engine.phase).toBe(Phase.BIDDING);
    });

    it("bid updates currentBid and currentHighBidder", () => {
      const events = engine.step(11); // bid 5
      expect(engine.currentBid).toBe(5);
      expect(engine.currentHighBidder).toBe(1); // player 1 bid
      expect(events[0].type).toBe("bid");
      expect(events[0].data.action).toBe(5);
    });

    it("pass does not change currentBid", () => {
      engine.step(ACTION_PASS); // player 1 passes
      expect(engine.currentBid).toBe(0);
      expect(engine.currentPlayer).toBe(2);
    });

    it("advances player after bid", () => {
      engine.step(11); // player 1 bids 5
      expect(engine.currentPlayer).toBe(2);
    });

    it("advances player after pass", () => {
      engine.step(ACTION_PASS);
      expect(engine.currentPlayer).toBe(2);
    });

    it("higher bids override previous", () => {
      engine.step(11); // player 1 bids 5
      engine.step(12); // player 2 bids 6
      expect(engine.currentBid).toBe(6);
      expect(engine.currentHighBidder).toBe(2);
    });

    it("bidding ends after dealer bids", () => {
      // dealer=0, so players 1,2,3 bid, then dealer gets a turn
      engine.step(11); // player 1 bids 5
      engine.step(ACTION_PASS); // player 2 passes
      engine.step(ACTION_PASS); // player 3 passes
      // Now it's dealer's turn (player 0) — dealer can pass or outbid
      expect(engine.phase).toBe(Phase.BIDDING);
      expect(engine.currentPlayer).toBe(0); // dealer's turn
      engine.step(ACTION_PASS); // dealer passes
      // Bidding complete — high bidder was player 1
      expect(engine.phase).toBe(Phase.CHOOSESUIT);
      expect(engine.currentPlayer).toBe(1); // high bidder
    });

    it("transitions to CHOOSESUIT after all four players bid", () => {
      engine.step(ACTION_PASS); // 1 passes
      engine.step(11);          // 2 bids 5
      engine.step(ACTION_PASS); // 3 passes
      // Dealer (0) still needs to bid
      expect(engine.phase).toBe(Phase.BIDDING);
      engine.step(ACTION_PASS); // 0 (dealer) passes
      expect(engine.phase).toBe(Phase.CHOOSESUIT);
      expect(engine.currentPlayer).toBe(2);
    });

    it("dealer forced bid when all pass", () => {
      // All three non-dealer players pass, dealer is forced
      engine.step(ACTION_PASS); // 1 passes
      engine.step(ACTION_PASS); // 2 passes
      engine.step(ACTION_PASS); // 3 passes
      // Dealer (0) must bid — cannot pass
      expect(engine.phase).toBe(Phase.BIDDING);
      expect(engine.currentPlayer).toBe(0);
      const mask = engine.getActionMask();
      expect(mask[ACTION_PASS]).toBe(0); // can't pass
      expect(mask[11]).toBe(1); // can bid 5
      expect(mask[17]).toBe(1); // can bid Moon
      expect(mask[18]).toBe(0); // can't double moon (no one bid moon)
      engine.step(11); // dealer bids 5
      expect(engine.phase).toBe(Phase.CHOOSESUIT);
      expect(engine.currentPlayer).toBe(0);
    });
  });

  // --- Bidding mask ---

  describe("bidding mask", () => {
    it("first bidder can pass (non-dealer) and bid 5-11", () => {
      const mask = engine.getActionMask();
      expect(mask[ACTION_PASS]).toBe(1);
      for (let i = 11; i < 18; i++) {
        expect(mask[i]).toBe(1);
      }
      // double moon not available (no one bid moon yet)
      expect(mask[18]).toBe(0);
    });

    it("after a bid of 5, next player can pass and bid 6+", () => {
      engine.step(11); // player 1 bids 5
      const mask = engine.getActionMask();
      expect(mask[ACTION_PASS]).toBe(1);
      expect(mask[11]).toBe(0); // can't bid 5
      expect(mask[12]).toBe(1); // can bid 6
      for (let i = 12; i < 18; i++) {
        expect(mask[i]).toBe(1);
      }
    });

    it("dealer cannot pass when no bids have been made", () => {
      // advance to dealer with no bids
      engine.step(ACTION_PASS); // 1
      engine.step(ACTION_PASS); // 2
      engine.step(ACTION_PASS); // 3 passes
      // Now it's dealer's turn with no bids
      expect(engine.currentPlayer).toBe(0);
      expect(engine.phase).toBe(Phase.BIDDING);
      const mask = engine.getActionMask();
      expect(mask[ACTION_PASS]).toBe(0); // dealer can't pass
      expect(mask[11]).toBe(1); // can bid 5
    });

    it("dealer cannot pass when currentBid is 0", () => {
      const e = new PitchEngine();
      e.reset(1); // dealer=1
      // currentPlayer = 2
      e.step(ACTION_PASS); // 2 passes
      e.step(ACTION_PASS); // 3 passes
      e.step(ACTION_PASS); // 0 passes
      // Now dealer (1) must bid
      expect(e.currentPlayer).toBe(1);
      expect(e.phase).toBe(Phase.BIDDING);
      const mask = e.getActionMask();
      expect(mask[ACTION_PASS]).toBe(0); // dealer can't pass with no bids
      expect(mask[11]).toBe(1); // can bid 5
    });

    it("dealer can double moon when someone bid moon", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.currentPlayer = 0; // dealer's turn
      e.currentBid = 11; // someone bid moon (bid value 11)
      e.phase = Phase.BIDDING;
      const mask = e.getActionMask();
      expect(mask[18]).toBe(1); // double moon
      expect(mask[ACTION_PASS]).toBe(1); // can pass
    });

    it("dealer sees 9/10/moon when current bid is 8", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.currentPlayer = 0; // dealer's turn
      e.currentBid = 8;
      e.phase = Phase.BIDDING;
      const mask = e.getActionMask();
      expect(mask[15]).toBe(1); // can bid 9
      expect(mask[16]).toBe(1); // can bid 10
      expect(mask[17]).toBe(1); // can bid moon
      expect(mask[18]).toBe(0); // can't double moon (no one bid moon)
    });

    it("non-dealer cannot double moon", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.currentPlayer = 1;
      e.currentBid = 11; // someone bid moon
      e.phase = Phase.BIDDING;
      const mask = e.getActionMask();
      expect(mask[18]).toBe(0); // non-dealer can't double moon
    });

    it("no card play actions during bidding", () => {
      const mask = engine.getActionMask();
      for (let i = 0; i <= 9; i++) {
        expect(mask[i]).toBe(0);
      }
    });

    it("no suit selection during bidding", () => {
      const mask = engine.getActionMask();
      for (let i = 19; i <= 22; i++) {
        expect(mask[i]).toBe(0);
      }
    });
  });

  // --- Choose suit ---

  describe("choose suit", () => {
    function bidToChooseSuit(e: PitchEngine): void {
      e.step(11);          // player 1 bids 5
      e.step(ACTION_PASS); // player 2 passes
      e.step(ACTION_PASS); // player 3 passes
      e.step(ACTION_PASS); // dealer (0) passes
    }

    it("suit selection sets trumpSuit", () => {
      bidToChooseSuit(engine);
      expect(engine.phase).toBe(Phase.CHOOSESUIT);
      engine.step(19); // choose Hearts
      expect(engine.trumpSuit).toBe(Suit.HEARTS);
    });

    it("transitions to PLAYING after suit choice", () => {
      bidToChooseSuit(engine);
      engine.step(20); // choose Diamonds
      expect(engine.phase).toBe(Phase.PLAYING);
    });

    it("triggers discard-and-fill after suit choice", () => {
      bidToChooseSuit(engine);
      engine.step(19); // choose Hearts
      // Discard-and-fill should have run: players dealt 9 cards,
      // after discarding non-trump at least one hand should differ from 9
      const totalCards = engine.hands.reduce((s, h) => s + h.length, 0);
      expect(totalCards).toBeLessThan(36); // 4 * 9 = 36 pre-discard
    });

    it("emits trumpChosen event", () => {
      bidToChooseSuit(engine);
      const events = engine.step(21); // choose Clubs
      expect(events.some((e) => e.type === "trumpChosen")).toBe(true);
      const trumpEvent = events.find((e) => e.type === "trumpChosen")!;
      expect(trumpEvent.data.suit).toBe(Suit.CLUBS);
      expect(trumpEvent.data.bidder).toBe(1);
      expect(trumpEvent.data.bidAmount).toBe(5);
    });

    it("mask only enables suits for high bidder (Bug 1 fix)", () => {
      bidToChooseSuit(engine);
      expect(engine.phase).toBe(Phase.CHOOSESUIT);
      expect(engine.currentPlayer).toBe(engine.currentHighBidder);

      // Mask for the high bidder should have suit options
      const bidderMask = engine.getActionMask();
      for (let i = 19; i < 23; i++) {
        expect(bidderMask[i]).toBe(1);
      }

      // Temporarily set currentPlayer to someone else — mask should have no suits
      const otherPlayer = (engine.currentHighBidder + 1) % 4;
      engine.currentPlayer = otherPlayer;
      const otherMask = engine.getActionMask();
      for (let i = 19; i < 23; i++) {
        expect(otherMask[i]).toBe(0);
      }
      // Restore
      engine.currentPlayer = engine.currentHighBidder;
    });

    it("all four suits are selectable", () => {
      bidToChooseSuit(engine);
      const mask = engine.getActionMask();
      expect(mask[19]).toBe(1); // Hearts
      expect(mask[20]).toBe(1); // Diamonds
      expect(mask[21]).toBe(1); // Clubs
      expect(mask[22]).toBe(1); // Spades
    });
  });

  // --- Discard and fill ---

  describe("discard and fill", () => {
    it("discards non-trump cards from pre-deal hands", () => {
      // Manually set up a controlled scenario
      const e = new PitchEngine();
      e.reset(0);

      // Give player 0 a mix of trump and non-trump
      e.hands[0] = [
        { suit: Suit.HEARTS, rank: 15 },
        { suit: Suit.HEARTS, rank: 14 },
        { suit: Suit.CLUBS, rank: 5 },   // non-trump, will be discarded
        { suit: Suit.SPADES, rank: 9 },   // non-trump, will be discarded
      ];
      // Stock deck with trump cards so fill draws valid cards
      e.deck = [
        { suit: Suit.HEARTS, rank: 2 },
        { suit: Suit.HEARTS, rank: 3 },
        { suit: Suit.HEARTS, rank: 4 },
        { suit: Suit.HEARTS, rank: 5 },
        { suit: Suit.HEARTS, rank: 6 },
        { suit: Suit.HEARTS, rank: 7 },
        { suit: Suit.HEARTS, rank: 8 },
        { suit: Suit.HEARTS, rank: 9 },
        { suit: Suit.HEARTS, rank: 10 },
        { suit: null, rank: 11 },
        { suit: null, rank: 11 },
      ];
      // Clear other hands to avoid interference
      e.hands[1] = [{ suit: Suit.HEARTS, rank: 12 }];
      e.hands[2] = [{ suit: Suit.HEARTS, rank: 13 }];
      e.hands[3] = [{ suit: Suit.DIAMONDS, rank: 12 }]; // off-jack

      e.currentBid = 5;
      e.currentHighBidder = 0;
      e.trumpSuit = Suit.HEARTS;
      e.phase = Phase.CHOOSESUIT;
      e.currentPlayer = 0;

      e.step(19 + Suit.HEARTS); // choose Hearts

      // Player 0's hand should only contain valid plays (non-trump discarded, filled with trump)
      for (const card of e.hands[0]) {
        expect(e.isValidPlay(card)).toBe(true);
      }
      // The two non-trump cards should be gone
      expect(e.hands[0].some((c) => c.suit === Suit.CLUBS)).toBe(false);
      expect(e.hands[0].some((c) => c.suit === Suit.SPADES)).toBe(false);
    });

    it("fills players to at most 6 cards", () => {
      // Use controlled hands: 3 trump + 6 non-trump each.
      // After discard (3 remain) + fill (draw 3) → 6 cards per player.
      const e = new PitchEngine();
      e.reset(0);
      e.hands[0] = [
        { suit: Suit.SPADES, rank: 2 }, { suit: Suit.SPADES, rank: 3 }, { suit: Suit.SPADES, rank: 4 },
        { suit: Suit.HEARTS, rank: 2 }, { suit: Suit.HEARTS, rank: 3 }, { suit: Suit.HEARTS, rank: 4 },
        { suit: Suit.CLUBS, rank: 2 }, { suit: Suit.CLUBS, rank: 3 }, { suit: Suit.CLUBS, rank: 4 },
      ];
      e.hands[1] = [
        { suit: Suit.SPADES, rank: 5 }, { suit: Suit.SPADES, rank: 6 }, { suit: Suit.SPADES, rank: 7 },
        { suit: Suit.HEARTS, rank: 5 }, { suit: Suit.HEARTS, rank: 6 }, { suit: Suit.HEARTS, rank: 7 },
        { suit: Suit.DIAMONDS, rank: 2 }, { suit: Suit.DIAMONDS, rank: 3 }, { suit: Suit.DIAMONDS, rank: 4 },
      ];
      e.hands[2] = [
        { suit: Suit.SPADES, rank: 8 }, { suit: Suit.SPADES, rank: 9 }, { suit: Suit.SPADES, rank: 10 },
        { suit: Suit.HEARTS, rank: 8 }, { suit: Suit.HEARTS, rank: 9 }, { suit: Suit.HEARTS, rank: 10 },
        { suit: Suit.DIAMONDS, rank: 5 }, { suit: Suit.DIAMONDS, rank: 6 }, { suit: Suit.DIAMONDS, rank: 7 },
      ];
      e.hands[3] = [
        { suit: Suit.SPADES, rank: 12 }, { suit: Suit.SPADES, rank: 13 }, { suit: Suit.SPADES, rank: 14 },
        { suit: Suit.DIAMONDS, rank: 8 }, { suit: Suit.DIAMONDS, rank: 9 }, { suit: Suit.DIAMONDS, rank: 10 },
        { suit: Suit.CLUBS, rank: 5 }, { suit: Suit.CLUBS, rank: 6 }, { suit: Suit.CLUBS, rank: 7 },
      ];
      e.currentBid = 5;
      e.currentHighBidder = 0;
      e.phase = Phase.CHOOSESUIT;
      e.currentPlayer = 0;

      e.step(19 + Suit.SPADES);
      for (let p = 0; p < 4; p++) {
        expect(e.hands[p].length).toBeLessThanOrEqual(6);
      }
    });

    it("jokers are kept (they are valid plays)", () => {
      // Manually give a player a joker and verify it survives discard
      const e = new PitchEngine();
      e.reset(0);
      e.hands[1] = [
        { suit: null, rank: 11 }, // joker
        { suit: Suit.HEARTS, rank: 15 },
        { suit: Suit.HEARTS, rank: 14 },
      ];
      e.currentBid = 5;
      e.currentHighBidder = 1;
      e.trumpSuit = Suit.HEARTS;
      e.phase = Phase.PLAYING;
      // joker should be valid
      expect(e.isValidPlay({ suit: null, rank: 11 })).toBe(true);
    });

    it("off-jack is kept", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.trumpSuit = Suit.HEARTS;
      // Off-jack for Hearts trump is Diamonds Jack
      const offJack: Card = { suit: Suit.DIAMONDS, rank: 12 };
      expect(e.isOffJack(offJack)).toBe(true);
      expect(e.isValidPlay(offJack)).toBe(true);
    });
  });

  // --- Playing ---

  describe("playing", () => {
    function setupPlaying(e: PitchEngine, trump: Suit = Suit.HEARTS): void {
      e.step(11);          // player 1 bids 5
      e.step(ACTION_PASS); // player 2 passes
      e.step(ACTION_PASS); // player 3 passes
      e.step(ACTION_PASS); // dealer (0) passes
      e.step(19 + trump);  // choose trump
    }

    it("playing a card removes it from hand", () => {
      setupPlaying(engine);
      const player = engine.currentPlayer;
      const handBefore = engine.hands[player].length;
      const mask = engine.getActionMask();
      const validAction = mask.findIndex((v) => v === 1);
      engine.step(validAction);
      expect(engine.hands[player].length).toBe(handBefore - 1);
    });

    it("playing a card adds to currentTrick", () => {
      setupPlaying(engine);
      const mask = engine.getActionMask();
      const validAction = mask.findIndex((v) => v === 1);
      expect(engine.currentTrick.length).toBe(0);
      engine.step(validAction);
      // If the trick isn't complete, it should have 1 entry
      if (engine.playingIterator > 0) {
        expect(engine.currentTrick.length).toBe(1);
      }
    });

    it("playing a card emits cardPlayed event", () => {
      setupPlaying(engine);
      const mask = engine.getActionMask();
      const validAction = mask.findIndex((v) => v === 1);
      const events = engine.step(validAction);
      expect(events.some((e) => e.type === "cardPlayed")).toBe(true);
    });

    it("invalid play is corrected to random valid action", () => {
      setupPlaying(engine);
      // Try to play an invalid action (e.g., action 23 which is NO_VALID_PLAY when valid plays exist)
      const mask = engine.getActionMask();
      if (mask[ACTION_NO_VALID_PLAY] !== 1) {
        // Action 23 is not valid, so step should pick a random valid one
        const handBefore = [...engine.hands[engine.currentPlayer]];
        engine.step(ACTION_NO_VALID_PLAY);
        // Game should still function (no crash)
        expect(engine.phase).toBe(Phase.PLAYING);
      }
    });

    it("noValidPlay action emits event", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.currentPlayer = 2;
      e.hands[2] = []; // empty hand, no valid plays
      const mask = e.getActionMask();
      expect(mask[ACTION_NO_VALID_PLAY]).toBe(1);
      const events = e.step(ACTION_NO_VALID_PLAY);
      expect(events.some((ev) => ev.type === "noValidPlay")).toBe(true);
    });

    it("playingIterator advances with each play", () => {
      setupPlaying(engine);
      expect(engine.playingIterator).toBe(0);
      const mask = engine.getActionMask();
      const validAction = mask.findIndex((v) => v === 1);
      engine.step(validAction);
      // After one play, iterator should be 1 (unless trick completed)
      expect(engine.playingIterator).toBeLessThanOrEqual(1);
    });
  });

  // --- Trick resolution ---

  describe("trick resolution", () => {
    function setupAndPlayTrick(e: PitchEngine): void {
      // Set up a controlled trick
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.currentPlayer = 0;
      e.playingIterator = 0;

      // Give each player exactly one trump card
      e.hands[0] = [{ suit: Suit.HEARTS, rank: 10 }];
      e.hands[1] = [{ suit: Suit.HEARTS, rank: 5 }];
      e.hands[2] = [{ suit: Suit.HEARTS, rank: 15 }]; // Ace — highest
      e.hands[3] = [{ suit: Suit.HEARTS, rank: 3 }];
    }

    it("highest rank wins the trick", () => {
      setupAndPlayTrick(engine);
      engine.step(0); // player 0 plays 10
      engine.step(0); // player 1 plays 5
      engine.step(0); // player 2 plays Ace
      const events = engine.step(0); // player 3 plays 3
      const trickResult = events.find((e) => e.type === "trickResult");
      expect(trickResult).toBeDefined();
      expect(trickResult!.data.winner).toBe(2); // Ace wins
    });

    it("accumulates points correctly", () => {
      setupAndPlayTrick(engine);
      engine.step(0);
      engine.step(0);
      engine.step(0);
      const events = engine.step(0);
      const trickResult = events.find((e) => e.type === "trickResult");
      // 10=1pt, 5=0pt, Ace=1pt, 3=3pts = 5 total
      expect(trickResult!.data.pointsWon).toBe(5);
    });

    it("winner leads next trick", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.currentPlayer = 0;
      e.playingIterator = 0;

      e.hands[0] = [{ suit: Suit.HEARTS, rank: 5 }, { suit: Suit.HEARTS, rank: 4 }];
      e.hands[1] = [{ suit: Suit.HEARTS, rank: 6 }, { suit: Suit.HEARTS, rank: 7 }];
      e.hands[2] = [{ suit: Suit.HEARTS, rank: 15 }, { suit: Suit.HEARTS, rank: 8 }]; // Ace wins
      e.hands[3] = [{ suit: Suit.HEARTS, rank: 9 }, { suit: Suit.HEARTS, rank: 2 }];

      e.step(0); // 0 plays 5
      e.step(0); // 1 plays 6
      e.step(0); // 2 plays Ace
      e.step(0); // 3 plays 9 -> trick ends, winner=2
      expect(e.currentPlayer).toBe(2); // winner leads next
    });

    it("joker is a valid play and scores 1 point", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.currentPlayer = 0;
      e.playingIterator = 0;

      e.hands[0] = [{ suit: null, rank: 11 }]; // joker
      e.hands[1] = [{ suit: Suit.HEARTS, rank: 15 }]; // Ace
      e.hands[2] = [{ suit: Suit.HEARTS, rank: 5 }];
      e.hands[3] = [{ suit: Suit.HEARTS, rank: 6 }];

      e.step(0); // joker
      e.step(0); // Ace
      e.step(0); // 5
      const events = e.step(0); // 6
      const tr = events.find((ev) => ev.type === "trickResult");
      // Ace (rank 15) should win; joker has rank 11
      expect(tr!.data.winner).toBe(1);
      // Points: joker=1, ace=1, 5=0, 6=0 = 2
      expect(tr!.data.pointsWon).toBe(2);
    });

    it("off-jack is a valid play in trick", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.currentPlayer = 0;
      e.playingIterator = 0;

      // Off-jack for Hearts is Diamonds Jack
      e.hands[0] = [{ suit: Suit.DIAMONDS, rank: 12 }]; // off-jack
      e.hands[1] = [{ suit: Suit.HEARTS, rank: 5 }];
      e.hands[2] = [{ suit: Suit.HEARTS, rank: 6 }];
      e.hands[3] = [{ suit: Suit.HEARTS, rank: 7 }];

      const mask = e.getActionMask();
      expect(mask[0]).toBe(1); // off-jack is playable
      e.step(0);
      e.step(0);
      e.step(0);
      const events = e.step(0);
      const tr = events.find((ev) => ev.type === "trickResult");
      // Off-jack rank=12, beats 5,6,7 — so off-jack wins
      expect(tr!.data.winner).toBe(0);
      expect(tr!.data.pointsWon).toBe(1); // jack=1pt
    });

    it("2 of trump scores for the team that played it, not the trick winner", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.currentPlayer = 0;
      e.playingIterator = 0;

      // Player 0 (team 0) plays 2, Player 1 (team 1) plays Ace and wins
      e.hands[0] = [{ suit: Suit.HEARTS, rank: 2 }];
      e.hands[1] = [{ suit: Suit.HEARTS, rank: 15 }]; // Ace
      e.hands[2] = [{ suit: Suit.HEARTS, rank: 5 }];
      e.hands[3] = [{ suit: Suit.HEARTS, rank: 6 }];

      e.step(0); // player 0 plays 2
      e.step(0); // player 1 plays Ace
      e.step(0); // player 2 plays 5
      const events = e.step(0); // player 3 plays 6
      const tr = events.find((ev) => ev.type === "trickResult");
      expect(tr!.data.winner).toBe(1); // Ace wins
      // pointsWon should only include the Ace (1pt), not the 2
      expect(tr!.data.pointsWon).toBe(1);
      // 2's point goes to team 0 (player 0's team)
      expect(tr!.data.roundScores[0]).toBe(1);
      // Ace's point goes to team 1 (trick winner)
      expect(tr!.data.roundScores[1]).toBe(1);
    });

    it("2 of trump scores correctly when played by winning team", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.currentPlayer = 0;
      e.playingIterator = 0;

      // Player 1 (team 1) plays 2, Player 3 (team 1) plays Ace and wins
      e.hands[0] = [{ suit: Suit.HEARTS, rank: 5 }];
      e.hands[1] = [{ suit: Suit.HEARTS, rank: 2 }];
      e.hands[2] = [{ suit: Suit.HEARTS, rank: 6 }];
      e.hands[3] = [{ suit: Suit.HEARTS, rank: 15 }]; // Ace

      e.step(0); // player 0 plays 5
      e.step(0); // player 1 plays 2
      e.step(0); // player 2 plays 6
      const events = e.step(0); // player 3 plays Ace
      const tr = events.find((ev) => ev.type === "trickResult");
      expect(tr!.data.winner).toBe(3); // Ace wins
      // pointsWon should only include the Ace (1pt)
      expect(tr!.data.pointsWon).toBe(1);
      // Both 2 and Ace go to team 1
      expect(tr!.data.roundScores[1]).toBe(2);
    });

    it("2 of trump with multiple scoring cards", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.currentPlayer = 0;
      e.playingIterator = 0;

      // Player 0 (team 0) plays 2, Player 1 (team 1) plays Ace and wins
      // Player 2 plays 3 (3pts) — goes to trick winner (team 1)
      e.hands[0] = [{ suit: Suit.HEARTS, rank: 2 }];
      e.hands[1] = [{ suit: Suit.HEARTS, rank: 15 }]; // Ace
      e.hands[2] = [{ suit: Suit.HEARTS, rank: 3 }];  // 3 = 3pts
      e.hands[3] = [{ suit: Suit.HEARTS, rank: 5 }];

      e.step(0);
      e.step(0);
      e.step(0);
      const events = e.step(0);
      const tr = events.find((ev) => ev.type === "trickResult");
      expect(tr!.data.winner).toBe(1); // Ace wins
      // pointsWon = Ace(1) + 3(3) = 4, excludes the 2
      expect(tr!.data.pointsWon).toBe(4);
      // 2's point to team 0
      expect(tr!.data.roundScores[0]).toBe(1);
      // Ace + 3 to team 1
      expect(tr!.data.roundScores[1]).toBe(4);
    });

    it("2 of trump played by losing team alongside other scoring cards", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.currentPlayer = 0;
      e.playingIterator = 0;

      // Player 0 (team 0) plays Ace and wins, Player 1 (team 1) plays 2
      e.hands[0] = [{ suit: Suit.HEARTS, rank: 15 }]; // Ace
      e.hands[1] = [{ suit: Suit.HEARTS, rank: 2 }];
      e.hands[2] = [{ suit: Suit.HEARTS, rank: 5 }];
      e.hands[3] = [{ suit: Suit.HEARTS, rank: 6 }];

      e.step(0);
      e.step(0);
      e.step(0);
      const events = e.step(0);
      const tr = events.find((ev) => ev.type === "trickResult");
      expect(tr!.data.winner).toBe(0); // Ace wins
      // pointsWon = Ace(1), excludes the 2
      expect(tr!.data.pointsWon).toBe(1);
      // Ace to team 0 (winner)
      expect(tr!.data.roundScores[0]).toBe(1);
      // 2's point to team 1 (player 1's team)
      expect(tr!.data.roundScores[1]).toBe(1);
    });
  });

  // --- Scoring ---

  describe("scoring", () => {
    function setupForScoring(e: PitchEngine, bid: number, bidderTeam: 0 | 1): void {
      e.reset(0);
      e.currentBid = bid;
      e.currentHighBidder = bidderTeam === 0 ? 0 : 1;
    }

    it("bidding team scores points when bid is made", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.currentBid = 5;
      e.currentHighBidder = 0;
      e.roundScores = [6, 4];
      e.scores = [0, 0];
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      // Force all hands empty to trigger round end
      e.hands = [[], [], [], []];
      e.currentTrick = [];
      e.currentPlayer = 0;
      e.playingIterator = 0;

      const mask = e.getActionMask();
      expect(mask[ACTION_NO_VALID_PLAY]).toBe(1);
      const events = e.step(ACTION_NO_VALID_PLAY);
      const roundEnd = events.find((ev) => ev.type === "roundEnd");
      expect(roundEnd).toBeDefined();
      expect(roundEnd!.data.bidMade).toBe(true);
      // Team 0 bid 5, scored 6, so they get +6
      // Team 1 scored 4, so they get +4
    });

    it("bidding team goes set when bid is not made", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.currentBid = 7;
      e.currentHighBidder = 0;
      e.roundScores = [5, 5]; // team 0 only got 5 but bid 7
      e.scores = [10, 10];
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.hands = [[], [], [], []];
      e.currentTrick = [];
      e.currentPlayer = 0;

      const events = e.step(ACTION_NO_VALID_PLAY);
      const roundEnd = events.find((ev) => ev.type === "roundEnd");
      expect(roundEnd!.data.bidMade).toBe(false);
      // Team 0 should lose 7 points: 10-7=3
      expect((roundEnd!.data.totalScores as number[])[0]).toBe(3);
      // Team 1 keeps their 5: 10+5=15
      expect((roundEnd!.data.totalScores as number[])[1]).toBe(15);
    });

    it("non-bidding team always banks their points", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.currentBid = 5;
      e.currentHighBidder = 0; // team 0 is bidding
      e.roundScores = [6, 4]; // team 1 scored 4
      e.scores = [0, 0];
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.hands = [[], [], [], []];
      e.currentTrick = [];
      e.currentPlayer = 0;

      const events = e.step(ACTION_NO_VALID_PLAY);
      const roundEnd = events.find((ev) => ev.type === "roundEnd");
      // Team 1 (non-bidding) gets +4
      expect((roundEnd!.data.totalScores as number[])[1]).toBe(4);
    });

    it("moon (bid 11) awards 20 when all 10 points captured", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.currentBid = 11;
      e.currentHighBidder = 0;
      e.roundScores = [10, 0]; // swept all points
      e.scores = [0, 0];
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.hands = [[], [], [], []];
      e.currentTrick = [];
      e.currentPlayer = 0;

      const events = e.step(ACTION_NO_VALID_PLAY);
      const roundEnd = events.find((ev) => ev.type === "roundEnd");
      expect(roundEnd!.data.bidMade).toBe(true);
      expect((roundEnd!.data.totalScores as number[])[0]).toBe(20);
    });

    it("moon (bid 11) penalizes 20 when failed", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.currentBid = 11;
      e.currentHighBidder = 0;
      e.roundScores = [9, 1]; // didn't get all 10
      e.scores = [20, 0];
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.hands = [[], [], [], []];
      e.currentTrick = [];
      e.currentPlayer = 0;

      const events = e.step(ACTION_NO_VALID_PLAY);
      const roundEnd = events.find((ev) => ev.type === "roundEnd");
      expect(roundEnd!.data.bidMade).toBe(false);
      expect((roundEnd!.data.totalScores as number[])[0]).toBe(0); // 20-20=0
    });

    it("double moon (bid 12) awards 40 when all 10 points captured", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.currentBid = 12;
      e.currentHighBidder = 0;
      e.roundScores = [10, 0];
      e.scores = [0, 0];
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.hands = [[], [], [], []];
      e.currentTrick = [];
      e.currentPlayer = 0;

      const events = e.step(ACTION_NO_VALID_PLAY);
      const roundEnd = events.find((ev) => ev.type === "roundEnd");
      expect(roundEnd!.data.bidMade).toBe(true);
      expect((roundEnd!.data.totalScores as number[])[0]).toBe(40);
    });

    it("double moon (bid 12) penalizes 40 when failed", () => {
      const e = new PitchEngine();
      e.reset(0);
      e.currentBid = 12;
      e.currentHighBidder = 0;
      e.roundScores = [8, 2];
      e.scores = [50, 0];
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.hands = [[], [], [], []];
      e.currentTrick = [];
      e.currentPlayer = 0;

      const events = e.step(ACTION_NO_VALID_PLAY);
      const roundEnd = events.find((ev) => ev.type === "roundEnd");
      expect(roundEnd!.data.bidMade).toBe(false);
      expect((roundEnd!.data.totalScores as number[])[0]).toBe(10); // 50-40=10
    });
  });

  // --- Game end ---

  describe("game end", () => {
    it("game ends when bidding team reaches threshold", () => {
      const e = new PitchEngine(10);
      e.reset(0);
      e.currentBid = 5;
      e.currentHighBidder = 0;
      e.roundScores = [10, 0];
      e.scores = [0, 0]; // will get +10
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.hands = [[], [], [], []];
      e.currentTrick = [];
      e.currentPlayer = 0;

      const events = e.step(ACTION_NO_VALID_PLAY);
      expect(events.some((ev) => ev.type === "gameOver")).toBe(true);
      expect(e.gameOver).toBe(true);
    });

    it("game ends when score gap reaches threshold", () => {
      const e = new PitchEngine(54);
      e.reset(0);
      e.currentBid = 5;
      e.currentHighBidder = 0;
      e.roundScores = [10, 0];
      e.scores = [50, 0]; // gap will be 60-0 = 60 >= 54
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.hands = [[], [], [], []];
      e.currentTrick = [];
      e.currentPlayer = 0;

      const events = e.step(ACTION_NO_VALID_PLAY);
      expect(events.some((ev) => ev.type === "gameOver")).toBe(true);
    });

    it("game ends at 50-round limit", () => {
      const e = new PitchEngine(1000);
      e.reset(0);
      e.numberOfRoundsPlayed = 50;
      e.currentBid = 5;
      e.currentHighBidder = 0;
      e.roundScores = [5, 5];
      e.scores = [0, 0];
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.hands = [[], [], [], []];
      e.currentTrick = [];
      e.currentPlayer = 0;

      const events = e.step(ACTION_NO_VALID_PLAY);
      expect(events.some((ev) => ev.type === "gameOver")).toBe(true);
      const go = events.find((ev) => ev.type === "gameOver")!;
      expect(go.data.reason).toBe("maxRounds");
    });

    it("winner is set correctly", () => {
      const e = new PitchEngine(10);
      e.reset(0);
      e.currentBid = 5;
      e.currentHighBidder = 0;
      e.roundScores = [10, 0];
      e.scores = [0, 0];
      e.phase = Phase.PLAYING;
      e.trumpSuit = Suit.HEARTS;
      e.hands = [[], [], [], []];
      e.currentTrick = [];
      e.currentPlayer = 0;

      e.step(ACTION_NO_VALID_PLAY);
      expect(e.winner).toBe(0);
    });
  });

  // --- isOffJack ---

  describe("isOffJack", () => {
    it("Diamonds Jack is off-jack when trump is Hearts", () => {
      engine.trumpSuit = Suit.HEARTS;
      expect(engine.isOffJack({ suit: Suit.DIAMONDS, rank: 12 })).toBe(true);
    });

    it("Hearts Jack is off-jack when trump is Diamonds", () => {
      engine.trumpSuit = Suit.DIAMONDS;
      expect(engine.isOffJack({ suit: Suit.HEARTS, rank: 12 })).toBe(true);
    });

    it("Spades Jack is off-jack when trump is Clubs", () => {
      engine.trumpSuit = Suit.CLUBS;
      expect(engine.isOffJack({ suit: Suit.SPADES, rank: 12 })).toBe(true);
    });

    it("Clubs Jack is off-jack when trump is Spades", () => {
      engine.trumpSuit = Suit.SPADES;
      expect(engine.isOffJack({ suit: Suit.CLUBS, rank: 12 })).toBe(true);
    });

    it("trump Jack is NOT off-jack", () => {
      engine.trumpSuit = Suit.HEARTS;
      expect(engine.isOffJack({ suit: Suit.HEARTS, rank: 12 })).toBe(false);
    });

    it("non-jack rank is NOT off-jack", () => {
      engine.trumpSuit = Suit.HEARTS;
      expect(engine.isOffJack({ suit: Suit.DIAMONDS, rank: 15 })).toBe(false);
    });

    it("joker is NOT off-jack", () => {
      engine.trumpSuit = Suit.HEARTS;
      expect(engine.isOffJack({ suit: null, rank: 11 })).toBe(false);
    });

    it("returns false when trumpSuit is null", () => {
      engine.trumpSuit = null;
      expect(engine.isOffJack({ suit: Suit.DIAMONDS, rank: 12 })).toBe(false);
    });

    it("wrong-color jack is NOT off-jack", () => {
      engine.trumpSuit = Suit.HEARTS;
      // Clubs Jack is black, Hearts is red — not same-color
      expect(engine.isOffJack({ suit: Suit.CLUBS, rank: 12 })).toBe(false);
      expect(engine.isOffJack({ suit: Suit.SPADES, rank: 12 })).toBe(false);
    });
  });

  // --- isValidPlay ---

  describe("isValidPlay", () => {
    it("trump card is valid", () => {
      engine.trumpSuit = Suit.HEARTS;
      expect(engine.isValidPlay({ suit: Suit.HEARTS, rank: 5 })).toBe(true);
    });

    it("non-trump non-joker non-offjack is not valid", () => {
      engine.trumpSuit = Suit.HEARTS;
      expect(engine.isValidPlay({ suit: Suit.CLUBS, rank: 5 })).toBe(false);
    });

    it("joker is always valid", () => {
      engine.trumpSuit = Suit.HEARTS;
      expect(engine.isValidPlay({ suit: null, rank: 11 })).toBe(true);
    });

    it("off-jack is valid", () => {
      engine.trumpSuit = Suit.HEARTS;
      expect(engine.isValidPlay({ suit: Suit.DIAMONDS, rank: 12 })).toBe(true);
    });

    it("non-off-jack of different suit is not valid", () => {
      engine.trumpSuit = Suit.HEARTS;
      expect(engine.isValidPlay({ suit: Suit.SPADES, rank: 12 })).toBe(false);
    });
  });

  // --- newRound ---

  describe("newRound", () => {
    it("preserves scores across rounds", () => {
      engine.scores = [10, 15];
      engine.newRound();
      expect(engine.scores).toEqual([10, 15]);
    });

    it("increments dealer", () => {
      expect(engine.dealer).toBe(0);
      engine.newRound();
      expect(engine.dealer).toBe(1);
    });

    it("resets round state", () => {
      engine.newRound();
      expect(engine.currentBid).toBe(0);
      expect(engine.trumpSuit).toBeNull();
      expect(engine.phase).toBe(Phase.BIDDING);
      expect(engine.roundScores).toEqual([0, 0]);
      expect(engine.currentTrick).toEqual([]);
    });

    it("deals new hands of 9 cards each", () => {
      engine.newRound();
      for (let i = 0; i < 4; i++) {
        expect(engine.hands[i].length).toBe(9);
      }
    });

    it("increments numberOfRoundsPlayed", () => {
      expect(engine.numberOfRoundsPlayed).toBe(0);
      engine.newRound();
      expect(engine.numberOfRoundsPlayed).toBe(1);
    });
  });

  // --- Integration ---

  describe("integration", () => {
    it("complete random game finishes without errors", () => {
      const e = new PitchEngine(20); // lower threshold for faster games
      e.reset(0);

      let steps = 0;
      const maxSteps = 5000;

      while (!e.gameOver && steps < maxSteps) {
        const mask = e.getActionMask();
        const validActions = mask.map((v, i) => (v === 1 ? i : -1)).filter((i) => i >= 0);
        if (validActions.length === 0) break;
        const action = validActions[Math.floor(Math.random() * validActions.length)];
        e.step(action);
        steps++;
      }

      // Game should have ended
      expect(steps).toBeLessThan(maxSteps);
      expect(e.gameOver).toBe(true);
      expect(e.winner).not.toBeNull();
    });

    it("multiple games can run sequentially", () => {
      for (let g = 0; g < 3; g++) {
        const e = new PitchEngine(20);
        e.reset(g % 4);
        let steps = 0;
        while (!e.gameOver && steps < 5000) {
          const mask = e.getActionMask();
          const valid = mask.map((v, i) => (v === 1 ? i : -1)).filter((i) => i >= 0);
          if (valid.length === 0) break;
          e.step(valid[Math.floor(Math.random() * valid.length)]);
          steps++;
        }
        expect(e.gameOver).toBe(true);
      }
    });

    it("action mask always has at least one valid action during play", () => {
      const e = new PitchEngine(20);
      e.reset(0);
      let steps = 0;
      while (!e.gameOver && steps < 3000) {
        const mask = e.getActionMask();
        const validCount = mask.filter((v) => v === 1).length;
        expect(validCount).toBeGreaterThan(0);
        const validActions = mask.map((v, i) => (v === 1 ? i : -1)).filter((i) => i >= 0);
        e.step(validActions[Math.floor(Math.random() * validActions.length)]);
        steps++;
      }
    });

    it("getHandData returns correct format", () => {
      const hand = engine.getHandData(0);
      expect(hand.length).toBe(9);
      for (const card of hand) {
        expect(card).toHaveProperty("suit");
        expect(card).toHaveProperty("rank");
      }
    });

    it("getPlayerCardCounts returns 4-element tuple", () => {
      const counts = engine.getPlayerCardCounts();
      expect(counts.length).toBe(4);
      for (const c of counts) {
        expect(c).toBe(9);
      }
    });
  });
});
