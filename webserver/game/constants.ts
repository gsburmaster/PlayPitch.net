export enum Suit {
  HEARTS = 0,
  DIAMONDS = 1,
  CLUBS = 2,
  SPADES = 3,
}

export enum Phase {
  BIDDING = 0,
  CHOOSESUIT = 1,
  PLAYING = 2,
}

export interface Card {
  suit: Suit | null; // null for jokers
  rank: number; // 2-10 number cards, 11=Joker, 12=J, 13=Q, 14=K, 15=A
}

export interface CardData {
  suit: number | null; // 0-3 or null for jokers
  rank: number;
}

export function cardToData(card: Card): CardData {
  return { suit: card.suit !== null ? card.suit : null, rank: card.rank };
}

export function cardPoints(card: Card): number {
  if (card.rank === 15 || card.rank === 12 || card.rank === 10) return 1; // Ace, Jack, 10
  if (card.rank === 11) return 1; // Joker
  if (card.rank === 3) return 3;
  if (card.rank === 2) return 1;
  return 0;
}

export const RANK_NAMES: Record<number, string> = {
  11: "Joker",
  12: "J",
  13: "Q",
  14: "K",
  15: "A",
};

export const SUIT_NAMES: Record<number, string> = {
  [Suit.HEARTS]: "Hearts",
  [Suit.DIAMONDS]: "Diamonds",
  [Suit.CLUBS]: "Clubs",
  [Suit.SPADES]: "Spades",
};

// Off-jack pairs: same color, different suit
export const OFF_JACK_MAP: Record<Suit, Suit> = {
  [Suit.CLUBS]: Suit.SPADES,
  [Suit.SPADES]: Suit.CLUBS,
  [Suit.DIAMONDS]: Suit.HEARTS,
  [Suit.HEARTS]: Suit.DIAMONDS,
};

// Action encoding
export const ACTION_PASS = 10;
export const ACTION_BID_START = 11; // 11=bid5, 12=bid6, ..., 17=moon, 18=dblmoon
export const ACTION_SUIT_START = 19; // 19=hearts, 20=diamonds, 21=clubs, 22=spades
export const ACTION_NO_VALID_PLAY = 23;
export const NUM_ACTIONS = 24;
