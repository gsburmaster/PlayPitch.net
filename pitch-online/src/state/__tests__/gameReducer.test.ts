import { describe, it, expect } from "vitest";
import { gameReducer, initialGameState, type GameState } from "../gameReducer";
import type { CardData } from "../../types";

function stateWith(overrides: Partial<GameState>): GameState {
  return { ...initialGameState, ...overrides };
}

const card = (suit: 0 | 1 | 2 | 3 | null, rank: number): CardData => ({ suit, rank });

describe("gameReducer", () => {
  describe("SET_LOCAL_SEAT", () => {
    it("sets localSeat", () => {
      const next = gameReducer(initialGameState, { type: "SET_LOCAL_SEAT", seatIndex: 2 });
      expect(next.localSeat).toBe(2);
    });
  });

  describe("GAME_START", () => {
    it("initializes game state and preserves localSeat", () => {
      const hand = [card(0, 15), card(1, 10)];
      const state = stateWith({ localSeat: 1, scores: [10, 20] });
      const next = gameReducer(state, {
        type: "GAME_START",
        hand,
        dealer: 2,
        currentPlayer: 3,
        phase: 0,
        scores: [0, 0],
        roundNumber: 1,
        seats: [],
      });
      expect(next.myHand).toEqual(hand);
      expect(next.dealer).toBe(2);
      expect(next.currentPlayer).toBe(3);
      expect(next.localSeat).toBe(1);
      expect(next.playerCardCounts).toEqual([9, 9, 9, 9]);
      expect(next.scores).toEqual([0, 0]);
    });

    it("uses provided seats when non-empty", () => {
      const seats = [
        { seatIndex: 0 as const, displayName: "A", isAI: false, isConnected: true },
        { seatIndex: 1 as const, displayName: "B", isAI: true, isConnected: true },
        { seatIndex: 2 as const, displayName: "C", isAI: true, isConnected: true },
        { seatIndex: 3 as const, displayName: "D", isAI: true, isConnected: true },
      ];
      const next = gameReducer(initialGameState, {
        type: "GAME_START",
        hand: [],
        dealer: 0,
        currentPlayer: 0,
        phase: 0,
        scores: [0, 0],
        roundNumber: 1,
        seats,
      });
      expect(next.seats).toEqual(seats);
    });
  });

  describe("CARD_PLAYED", () => {
    const hand = [card(0, 15), card(1, 10), card(2, 12), card(null, 11)];

    it("removes card from hand when local player plays", () => {
      const state = stateWith({ localSeat: 0, myHand: hand, playerCardCounts: [4, 4, 4, 4] });
      const next = gameReducer(state, {
        type: "CARD_PLAYED",
        seatIndex: 0,
        card: card(1, 10),
        handIndex: 1,
      });
      expect(next.myHand).toHaveLength(3);
      expect(next.myHand).toEqual([card(0, 15), card(2, 12), card(null, 11)]);
    });

    it("does not remove card from hand when remote player plays", () => {
      const state = stateWith({ localSeat: 0, myHand: hand, playerCardCounts: [4, 4, 4, 4] });
      const next = gameReducer(state, {
        type: "CARD_PLAYED",
        seatIndex: 2,
        card: card(0, 15),
        handIndex: 0,
      });
      expect(next.myHand).toHaveLength(4);
      expect(next.myHand).toEqual(hand);
    });

    it("removes joker from hand when local player plays joker", () => {
      const state = stateWith({ localSeat: 0, myHand: hand, playerCardCounts: [4, 4, 4, 4] });
      const next = gameReducer(state, {
        type: "CARD_PLAYED",
        seatIndex: 0,
        card: card(null, 11),
        handIndex: 3,
      });
      expect(next.myHand).toHaveLength(3);
      expect(next.myHand).toEqual([card(0, 15), card(1, 10), card(2, 12)]);
    });

    it("does not remove local joker when remote player plays joker at same index", () => {
      const state = stateWith({ localSeat: 0, myHand: hand, playerCardCounts: [4, 4, 4, 4] });
      const next = gameReducer(state, {
        type: "CARD_PLAYED",
        seatIndex: 1,
        card: card(null, 11),
        handIndex: 3,
      });
      expect(next.myHand).toHaveLength(4);
      expect(next.myHand).toEqual(hand);
    });

    it("decrements playerCardCounts for the playing seat", () => {
      const state = stateWith({ localSeat: 0, myHand: hand, playerCardCounts: [4, 4, 4, 4] });
      const next = gameReducer(state, {
        type: "CARD_PLAYED",
        seatIndex: 2,
        card: card(0, 15),
        handIndex: 0,
      });
      expect(next.playerCardCounts).toEqual([4, 4, 3, 4]);
    });

    it("does not go below zero for playerCardCounts", () => {
      const state = stateWith({ localSeat: 0, myHand: [], playerCardCounts: [0, 0, 0, 0] });
      const next = gameReducer(state, {
        type: "CARD_PLAYED",
        seatIndex: 1,
        card: card(0, 15),
        handIndex: 0,
      });
      expect(next.playerCardCounts[1]).toBe(0);
    });

    it("adds card to currentTrick", () => {
      const state = stateWith({ localSeat: 0, myHand: hand, playerCardCounts: [4, 4, 4, 4] });
      const next = gameReducer(state, {
        type: "CARD_PLAYED",
        seatIndex: 2,
        card: card(0, 15),
        handIndex: 0,
      });
      expect(next.currentTrick).toHaveLength(1);
      expect(next.currentTrick[0]).toEqual({ card: card(0, 15), seatIndex: 2 });
    });
  });

  describe("HAND_UPDATE", () => {
    it("replaces hand and playerCardCounts", () => {
      const newHand = [card(0, 15), card(3, 10)];
      const next = gameReducer(initialGameState, {
        type: "HAND_UPDATE",
        hand: newHand,
        playerCardCounts: [2, 3, 3, 3],
      });
      expect(next.myHand).toEqual(newHand);
      expect(next.playerCardCounts).toEqual([2, 3, 3, 3]);
    });
  });

  describe("TURN_UPDATE", () => {
    it("sets turn state fields", () => {
      const mask = [10, 11, 12];
      const next = gameReducer(initialGameState, {
        type: "TURN_UPDATE",
        currentPlayer: 2,
        phase: 1,
        actionMask: mask,
        currentBid: 5,
        currentHighBidder: 1,
        trumpSuit: 3,
      });
      expect(next.currentPlayer).toBe(2);
      expect(next.phase).toBe(1);
      expect(next.actionMask).toEqual(mask);
      expect(next.currentBid).toBe(5);
      expect(next.currentHighBidder).toBe(1);
      expect(next.trumpSuit).toBe(3);
    });
  });

  describe("NEW_ROUND", () => {
    it("resets round state and sets new round fields", () => {
      const state = stateWith({
        localSeat: 1,
        trumpSuit: 2,
        currentBid: 7,
        bidHistory: [{ seatIndex: 0, action: 5, displayName: "A" }],
        currentTrick: [{ card: card(0, 15), seatIndex: 0 }],
        roundScores: [3, 1],
        lastTrickResult: { trick: [], winner: 0, winnerName: "A", pointsWon: 1, roundScores: [1, 0] },
      });
      const newHand = [card(1, 12)];
      const next = gameReducer(state, {
        type: "NEW_ROUND",
        hand: newHand,
        dealer: 1,
        currentPlayer: 2,
        roundNumber: 3,
        scores: [15, 20],
      });
      expect(next.myHand).toEqual(newHand);
      expect(next.dealer).toBe(1);
      expect(next.currentPlayer).toBe(2);
      expect(next.roundNumber).toBe(3);
      expect(next.scores).toEqual([15, 20]);
      expect(next.phase).toBe(0);
      expect(next.trumpSuit).toBeNull();
      expect(next.bidHistory).toEqual([]);
      expect(next.currentTrick).toEqual([]);
      expect(next.roundScores).toEqual([0, 0]);
      expect(next.lastTrickResult).toBeNull();
      expect(next.playerCardCounts).toEqual([9, 9, 9, 9]);
      expect(next.localSeat).toBe(1);
    });
  });

  describe("TRICK_RESULT / CLEAR_TRICK_RESULT", () => {
    it("sets lastTrickResult and roundScores", () => {
      const data = {
        trick: [{ card: card(0, 15), seatIndex: 0 as const }],
        winner: 0,
        winnerName: "A",
        pointsWon: 1,
        roundScores: [1, 0] as [number, number],
      };
      const next = gameReducer(initialGameState, { type: "TRICK_RESULT", data });
      expect(next.lastTrickResult).toEqual(data);
      expect(next.roundScores).toEqual([1, 0]);
    });

    it("CLEAR_TRICK_RESULT clears result and trick", () => {
      const state = stateWith({
        lastTrickResult: { trick: [], winner: 0, winnerName: "A", pointsWon: 0, roundScores: [0, 0] },
        currentTrick: [{ card: card(0, 15), seatIndex: 0 }],
      });
      const next = gameReducer(state, { type: "CLEAR_TRICK_RESULT" });
      expect(next.lastTrickResult).toBeNull();
      expect(next.currentTrick).toEqual([]);
    });

    it("CLEAR_TRICK_RESULT is no-op when lastTrickResult is already null", () => {
      const state = stateWith({
        lastTrickResult: null,
        currentTrick: [{ card: card(0, 15), seatIndex: 0 }],
      });
      const next = gameReducer(state, { type: "CLEAR_TRICK_RESULT" });
      expect(next).toBe(state);
      expect(next.currentTrick).toHaveLength(1);
    });
  });

  describe("CARD_PLAYED / CLEAR_TRICK_RESULT race condition", () => {
    const trickResult = {
      trick: [{ card: card(0, 15), seatIndex: 0 as const }],
      winner: 2,
      winnerName: "C",
      pointsWon: 1,
      roundScores: [1, 0] as [number, number],
    };

    it("CARD_PLAYED resets currentTrick and clears lastTrickResult when new trick starts", () => {
      const state = stateWith({
        lastTrickResult: trickResult,
        currentTrick: [
          { card: card(0, 15), seatIndex: 0 },
          { card: card(0, 10), seatIndex: 1 },
          { card: card(0, 12), seatIndex: 2 },
          { card: card(0, 14), seatIndex: 3 },
        ],
        playerCardCounts: [5, 5, 5, 5],
      });
      const next = gameReducer(state, {
        type: "CARD_PLAYED",
        seatIndex: 2,
        card: card(1, 15),
        handIndex: 0,
      });
      expect(next.currentTrick).toHaveLength(1);
      expect(next.currentTrick[0]).toEqual({ card: card(1, 15), seatIndex: 2 });
      expect(next.lastTrickResult).toBeNull();
    });

    it("CARD_PLAYED appends normally when lastTrickResult is null", () => {
      const state = stateWith({
        lastTrickResult: null,
        currentTrick: [{ card: card(0, 15), seatIndex: 0 }],
        playerCardCounts: [5, 5, 5, 5],
      });
      const next = gameReducer(state, {
        type: "CARD_PLAYED",
        seatIndex: 1,
        card: card(0, 10),
        handIndex: 0,
      });
      expect(next.currentTrick).toHaveLength(2);
      expect(next.lastTrickResult).toBeNull();
    });

    it("CLEAR_TRICK_RESULT after early CARD_PLAYED is no-op (new card preserved)", () => {
      // Simulate: trick result → winner plays fast → clear timer fires
      let state = stateWith({
        lastTrickResult: trickResult,
        currentTrick: [
          { card: card(0, 15), seatIndex: 0 },
          { card: card(0, 10), seatIndex: 1 },
          { card: card(0, 12), seatIndex: 2 },
          { card: card(0, 14), seatIndex: 3 },
        ],
        playerCardCounts: [5, 5, 5, 5],
      });
      // Winner plays at ~800ms
      state = gameReducer(state, {
        type: "CARD_PLAYED",
        seatIndex: 2,
        card: card(1, 15),
        handIndex: 0,
      });
      // Clear timer fires at 1500ms
      state = gameReducer(state, { type: "CLEAR_TRICK_RESULT" });
      // Winner's card must still be there
      expect(state.currentTrick).toHaveLength(1);
      expect(state.currentTrick[0]).toEqual({ card: card(1, 15), seatIndex: 2 });
    });

    it("multiple CARD_PLAYED before CLEAR_TRICK_RESULT preserves all new cards", () => {
      let state = stateWith({
        lastTrickResult: trickResult,
        currentTrick: [
          { card: card(0, 15), seatIndex: 0 },
          { card: card(0, 10), seatIndex: 1 },
          { card: card(0, 12), seatIndex: 2 },
          { card: card(0, 14), seatIndex: 3 },
        ],
        playerCardCounts: [5, 5, 5, 5],
      });
      // First new card (clears old trick)
      state = gameReducer(state, {
        type: "CARD_PLAYED",
        seatIndex: 2,
        card: card(1, 15),
        handIndex: 0,
      });
      // Second new card (appends normally)
      state = gameReducer(state, {
        type: "CARD_PLAYED",
        seatIndex: 3,
        card: card(1, 10),
        handIndex: 0,
      });
      // Clear timer fires — should be no-op
      state = gameReducer(state, { type: "CLEAR_TRICK_RESULT" });
      expect(state.currentTrick).toHaveLength(2);
      expect(state.currentTrick[0]).toEqual({ card: card(1, 15), seatIndex: 2 });
      expect(state.currentTrick[1]).toEqual({ card: card(1, 10), seatIndex: 3 });
    });
  });

  describe("BID", () => {
    it("appends to bidHistory", () => {
      const state = stateWith({
        bidHistory: [{ seatIndex: 0, action: 5, displayName: "A" }],
      });
      const next = gameReducer(state, {
        type: "BID",
        seatIndex: 1,
        action: "pass",
        displayName: "B",
      });
      expect(next.bidHistory).toHaveLength(2);
      expect(next.bidHistory[1]).toEqual({ seatIndex: 1, action: "pass", displayName: "B" });
    });
  });

  describe("TRUMP_CHOSEN", () => {
    it("sets trump and clears bid history", () => {
      const state = stateWith({
        bidHistory: [{ seatIndex: 0, action: 5, displayName: "A" }],
      });
      const next = gameReducer(state, {
        type: "TRUMP_CHOSEN",
        suit: 2,
        bidder: 0,
        bidderName: "A",
        bidAmount: 5,
      });
      expect(next.trumpSuit).toBe(2);
      expect(next.currentBid).toBe(5);
      expect(next.currentHighBidder).toBe(0);
      expect(next.bidHistory).toEqual([]);
    });
  });

  describe("RESET", () => {
    it("returns initial state", () => {
      const state = stateWith({ localSeat: 2, trumpSuit: 1, scores: [30, 40] });
      const next = gameReducer(state, { type: "RESET" });
      expect(next).toEqual(initialGameState);
    });
  });

  describe("ROUND_END", () => {
    it("sets roundEndData and updates scores", () => {
      const data = {
        bidderTeam: 0,
        bidAmount: 5,
        bidMade: true,
        roundScores: [4, 2] as [number, number],
        scoreDeltas: [5, 2] as [number, number],
        totalScores: [20, 12] as [number, number],
        newDealer: 1,
      };
      const next = gameReducer(initialGameState, { type: "ROUND_END", data });
      expect(next.roundEndData).toEqual(data);
      expect(next.scores).toEqual([20, 12]);
      expect(next.currentTrick).toEqual([]);
    });
  });

  describe("GAME_OVER", () => {
    it("sets gameOverData", () => {
      const data = { winner: 0 as const, finalScores: [54, 30] as [number, number], reason: "First to 54" };
      const next = gameReducer(initialGameState, { type: "GAME_OVER", data });
      expect(next.gameOverData).toEqual(data);
    });
  });

  describe("LOBBY_UPDATE", () => {
    it("updates seats", () => {
      const seats = [
        { seatIndex: 0 as const, displayName: "A", isAI: false, isConnected: true },
      ];
      const next = gameReducer(initialGameState, { type: "LOBBY_UPDATE", seats });
      expect(next.seats).toEqual(seats);
    });
  });

  describe("PLAYER_STATUS", () => {
    it("updates matching seat", () => {
      const seats = [
        { seatIndex: 0 as const, displayName: "A", isAI: false, isConnected: true },
        { seatIndex: 1 as const, displayName: "B", isAI: false, isConnected: true },
      ];
      const state = stateWith({ seats });
      const next = gameReducer(state, {
        type: "PLAYER_STATUS",
        seatIndex: 1,
        isConnected: false,
        displayName: "B2",
      });
      expect(next.seats[1].isConnected).toBe(false);
      expect(next.seats[1].displayName).toBe("B2");
      expect(next.seats[0]).toEqual(seats[0]);
    });
  });
});
