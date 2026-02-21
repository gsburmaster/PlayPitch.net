import type { WebSocket } from "ws";

export type SeatIndex = 0 | 1 | 2 | 3;

export interface SeatInfo {
  seatIndex: SeatIndex;
  displayName: string;
  isAI: boolean;
  isConnected: boolean;
}

export interface PlayerConnection {
  ws: WebSocket | null;
  playerId: string;
  displayName: string;
  seatIndex: SeatIndex;
  isAI: boolean;
  isConnected: boolean;
  disconnectTimer: ReturnType<typeof setTimeout> | null;
}

export interface CardData {
  suit: number | null;
  rank: number;
}

export interface TrickCard {
  card: CardData;
  seatIndex: SeatIndex;
}

// --- Client → Server messages ---
export interface AuthMessage {
  type: "auth";
  roomCode: string;
  playerId: string;
}

export interface GameActionMessage {
  type: "game:action";
  action: number;
}

export interface GameStartMessage {
  type: "game:start";
}

export interface RoomLeaveMessage {
  type: "room:leave";
}

export interface PlayAgainMessage {
  type: "room:playAgain";
}

export interface PingMessage {
  type: "ping";
}

export type ClientMessage =
  | AuthMessage
  | GameActionMessage
  | GameStartMessage
  | RoomLeaveMessage
  | PlayAgainMessage
  | PingMessage;

// --- Server → Client messages ---
export interface AuthOkMessage {
  type: "auth:ok";
  seatIndex: SeatIndex;
  roomCode: string;
  seats: SeatInfo[];
  isCreator: boolean;
}

export interface LobbyUpdateMessage {
  type: "lobby:update";
  seats: SeatInfo[];
}

export interface GameStartBroadcast {
  type: "game:start";
  hand: CardData[];
  dealer: number;
  currentPlayer: number;
  phase: number;
  scores: [number, number];
  roundNumber: number;
  aiModelLoaded: boolean;
}

export interface GameTurnMessage {
  type: "game:turn";
  currentPlayer: number;
  phase: number;
  actionMask: number[] | null;
  currentBid: number;
  currentHighBidder: number;
  trumpSuit: number | null;
}

export interface GameBidMessage {
  type: "game:bid";
  seatIndex: number;
  action: "pass" | number;
  displayName: string;
}

export interface GameTrumpChosenMessage {
  type: "game:trumpChosen";
  suit: number;
  bidder: number;
  bidderName: string;
  bidAmount: number;
}

export interface GameHandUpdateMessage {
  type: "game:handUpdate";
  hand: CardData[];
  playerCardCounts: [number, number, number, number];
}

export interface GameCardPlayedMessage {
  type: "game:cardPlayed";
  seatIndex: number;
  card: CardData;
  handIndex: number;
}

export interface GameNoValidPlayMessage {
  type: "game:noValidPlay";
  seatIndex: number;
}

export interface GameTrickResultMessage {
  type: "game:trickResult";
  trick: TrickCard[];
  winner: number;
  winnerName: string;
  pointsWon: number;
  roundScores: [number, number];
}

export interface GameRoundEndMessage {
  type: "game:roundEnd";
  bidderTeam: number;
  bidAmount: number;
  bidMade: boolean;
  roundScores: [number, number];
  scoreDeltas: [number, number];
  totalScores: [number, number];
  newDealer: number;
}

export interface GameNewRoundMessage {
  type: "game:newRound";
  hand: CardData[];
  dealer: number;
  currentPlayer: number;
  roundNumber: number;
  scores: [number, number];
}

export interface GameOverMessage {
  type: "game:over";
  winner: 0 | 1;
  finalScores: [number, number];
  reason: string;
}

export interface LobbyReturnMessage {
  type: "lobby:return";
  seats: SeatInfo[];
}

export interface PlayerStatusMessage {
  type: "player:status";
  seatIndex: number;
  isConnected: boolean;
  displayName: string;
}

export interface ErrorMessage {
  type: "error";
  message: string;
  code: string;
}

export interface PongMessage {
  type: "pong";
}

export type ServerMessage =
  | AuthOkMessage
  | LobbyUpdateMessage
  | GameStartBroadcast
  | GameTurnMessage
  | GameBidMessage
  | GameTrumpChosenMessage
  | GameHandUpdateMessage
  | GameCardPlayedMessage
  | GameNoValidPlayMessage
  | GameTrickResultMessage
  | GameRoundEndMessage
  | GameNewRoundMessage
  | GameOverMessage
  | LobbyReturnMessage
  | PlayerStatusMessage
  | ErrorMessage
  | PongMessage;
