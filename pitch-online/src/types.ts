export type Suit = 0 | 1 | 2 | 3;
export type Phase = 0 | 1 | 2;
export type SeatIndex = 0 | 1 | 2 | 3;

export interface CardData {
  suit: Suit | null;
  rank: number;
}

export interface SeatInfo {
  seatIndex: SeatIndex;
  displayName: string;
  isAI: boolean;
  isConnected: boolean;
}

export interface TrickCard {
  card: CardData;
  seatIndex: SeatIndex;
}

// View states
export type AppView = "splash" | "lobby" | "game";

// Suit display helpers
export const SUIT_SYMBOLS: Record<number, string> = {
  0: "\u2665", // Hearts
  1: "\u2666", // Diamonds
  2: "\u2663", // Clubs
  3: "\u2660", // Spades
};

export const SUIT_NAMES: Record<number, string> = {
  0: "Hearts",
  1: "Diamonds",
  2: "Clubs",
  3: "Spades",
};

export const SUIT_COLORS: Record<number, string> = {
  0: "red",
  1: "red",
  2: "black",
  3: "black",
};

export function getRankDisplay(rank: number): string {
  if (rank === 15) return "A";
  if (rank === 14) return "K";
  if (rank === 13) return "Q";
  if (rank === 12) return "J";
  if (rank === 11) return "JK";
  return String(rank);
}

export function getSuitSymbol(suit: number | null): string {
  if (suit === null) return "\u2605"; // Star for joker
  return SUIT_SYMBOLS[suit] ?? "";
}

export function getCardColor(suit: number | null): string {
  if (suit === null) return "#7b2d8b"; // Purple for joker
  return SUIT_COLORS[suit] ?? "black";
}

// Bid value to display
export function bidDisplay(bid: number): string {
  if (bid === 0) return "No bids";
  if (bid <= 10) return String(bid);
  if (bid === 11) return "Moon";
  if (bid === 12) return "Dbl Moon";
  return String(bid);
}
