import { createContext, useContext, useReducer, type Dispatch, type ReactNode } from "react";
import type { CardData, Phase, SeatIndex, SeatInfo, TrickCard } from "../types";

export interface BidEntry {
  seatIndex: number;
  action: "pass" | number;
  displayName: string;
}

export interface RoundEndData {
  bidderTeam: number;
  bidAmount: number;
  bidMade: boolean;
  roundScores: [number, number];
  scoreDeltas: [number, number];
  totalScores: [number, number];
  newDealer: number;
}

export interface GameOverData {
  winner: 0 | 1;
  finalScores: [number, number];
  reason: string;
}

export interface TrickResultData {
  trick: TrickCard[];
  winner: number;
  winnerName: string;
  pointsWon: number;
  roundScores: [number, number];
}

export interface GameState {
  seats: SeatInfo[];
  phase: Phase;
  dealer: number;
  currentPlayer: number;
  roundNumber: number;
  currentBid: number;
  currentHighBidder: number;
  bidHistory: BidEntry[];
  trumpSuit: number | null;
  myHand: CardData[];
  playerCardCounts: [number, number, number, number];
  actionMask: number[] | null;
  currentTrick: TrickCard[];
  scores: [number, number];
  roundScores: [number, number];
  lastTrickResult: TrickResultData | null;
  roundEndData: RoundEndData | null;
  gameOverData: GameOverData | null;
}

const initialGameState: GameState = {
  seats: [],
  phase: 0,
  dealer: 0,
  currentPlayer: 0,
  roundNumber: 0,
  currentBid: 0,
  currentHighBidder: 0,
  bidHistory: [],
  trumpSuit: null,
  myHand: [],
  playerCardCounts: [0, 0, 0, 0],
  actionMask: null,
  currentTrick: [],
  scores: [0, 0],
  roundScores: [0, 0],
  lastTrickResult: null,
  roundEndData: null,
  gameOverData: null,
};

export type GameAction =
  | { type: "GAME_START"; hand: CardData[]; dealer: number; currentPlayer: number; phase: Phase; scores: [number, number]; roundNumber: number; seats: SeatInfo[] }
  | { type: "TURN_UPDATE"; currentPlayer: number; phase: Phase; actionMask: number[] | null; currentBid: number; currentHighBidder: number; trumpSuit: number | null }
  | { type: "BID"; seatIndex: number; action: "pass" | number; displayName: string }
  | { type: "TRUMP_CHOSEN"; suit: number; bidder: number; bidderName: string; bidAmount: number }
  | { type: "HAND_UPDATE"; hand: CardData[]; playerCardCounts: [number, number, number, number] }
  | { type: "CARD_PLAYED"; seatIndex: number; card: CardData; handIndex: number }
  | { type: "NO_VALID_PLAY"; seatIndex: number }
  | { type: "TRICK_RESULT"; data: TrickResultData }
  | { type: "ROUND_END"; data: RoundEndData }
  | { type: "NEW_ROUND"; hand: CardData[]; dealer: number; currentPlayer: number; roundNumber: number; scores: [number, number] }
  | { type: "GAME_OVER"; data: GameOverData }
  | { type: "LOBBY_UPDATE"; seats: SeatInfo[] }
  | { type: "PLAYER_STATUS"; seatIndex: number; isConnected: boolean; displayName: string }
  | { type: "CLEAR_TRICK_RESULT" }
  | { type: "CLEAR_ROUND_END" }
  | { type: "RESET" };

function gameReducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case "GAME_START":
      return {
        ...initialGameState,
        seats: action.seats.length > 0 ? action.seats : state.seats,
        myHand: action.hand,
        dealer: action.dealer,
        currentPlayer: action.currentPlayer,
        phase: action.phase,
        scores: action.scores,
        roundNumber: action.roundNumber,
        playerCardCounts: [9, 9, 9, 9],
      };
    case "TURN_UPDATE":
      return {
        ...state,
        currentPlayer: action.currentPlayer,
        phase: action.phase,
        actionMask: action.actionMask,
        currentBid: action.currentBid,
        currentHighBidder: action.currentHighBidder,
        trumpSuit: action.trumpSuit,
      };
    case "BID":
      return {
        ...state,
        bidHistory: [...state.bidHistory, { seatIndex: action.seatIndex, action: action.action, displayName: action.displayName }],
      };
    case "TRUMP_CHOSEN":
      return {
        ...state,
        trumpSuit: action.suit,
        currentBid: action.bidAmount,
        currentHighBidder: action.bidder,
        bidHistory: [],
      };
    case "HAND_UPDATE":
      return {
        ...state,
        myHand: action.hand,
        playerCardCounts: action.playerCardCounts,
      };
    case "CARD_PLAYED":
      return {
        ...state,
        currentTrick: [...state.currentTrick, { card: action.card, seatIndex: action.seatIndex as SeatIndex }],
      };
    case "NO_VALID_PLAY":
      return state;
    case "TRICK_RESULT":
      return {
        ...state,
        lastTrickResult: action.data,
        roundScores: action.data.roundScores,
      };
    case "CLEAR_TRICK_RESULT":
      return { ...state, lastTrickResult: null, currentTrick: [] };
    case "ROUND_END":
      return {
        ...state,
        roundEndData: action.data,
        scores: action.data.totalScores,
        currentTrick: [],
      };
    case "CLEAR_ROUND_END":
      return { ...state, roundEndData: null };
    case "NEW_ROUND":
      return {
        ...state,
        myHand: action.hand,
        dealer: action.dealer,
        currentPlayer: action.currentPlayer,
        roundNumber: action.roundNumber,
        scores: action.scores,
        phase: 0,
        currentBid: 0,
        currentHighBidder: 0,
        bidHistory: [],
        trumpSuit: null,
        playerCardCounts: [9, 9, 9, 9],
        currentTrick: [],
        roundScores: [0, 0],
        lastTrickResult: null,
        roundEndData: null,
      };
    case "GAME_OVER":
      return { ...state, gameOverData: action.data };
    case "LOBBY_UPDATE":
      return { ...state, seats: action.seats };
    case "PLAYER_STATUS": {
      const seats = state.seats.map((s) =>
        s.seatIndex === action.seatIndex ? { ...s, isConnected: action.isConnected, displayName: action.displayName } : s
      );
      return { ...state, seats };
    }
    case "RESET":
      return initialGameState;
    default:
      return state;
  }
}

const GameContext = createContext<GameState>(initialGameState);
const GameDispatchContext = createContext<Dispatch<GameAction>>(() => {});

export function GameProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(gameReducer, initialGameState);
  return (
    <GameContext.Provider value={state}>
      <GameDispatchContext.Provider value={dispatch}>{children}</GameDispatchContext.Provider>
    </GameContext.Provider>
  );
}

export function useGameState() {
  return useContext(GameContext);
}

export function useGameDispatch() {
  return useContext(GameDispatchContext);
}
