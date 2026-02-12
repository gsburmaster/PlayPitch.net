export declare enum Suit {
    HEARTS = 0,
    DIAMONDS = 1,
    CLUBS = 2,
    SPADES = 3
}
export declare enum Phase {
    BIDDING = 0,
    CHOOSESUIT = 1,
    PLAYING = 2
}
export interface Card {
    suit: Suit | null;
    rank: number;
}
export interface CardData {
    suit: number | null;
    rank: number;
}
export declare function cardToData(card: Card): CardData;
export declare function cardPoints(card: Card): number;
export declare const RANK_NAMES: Record<number, string>;
export declare const SUIT_NAMES: Record<number, string>;
export declare const OFF_JACK_MAP: Record<Suit, Suit>;
export declare const ACTION_PASS = 10;
export declare const ACTION_BID_START = 11;
export declare const ACTION_SUIT_START = 19;
export declare const ACTION_NO_VALID_PLAY = 23;
export declare const NUM_ACTIONS = 24;
