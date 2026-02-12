import { Suit, Phase, Card, CardData } from "./constants.js";
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
export declare class PitchEngine {
    winThreshold: number;
    deck: Card[];
    hands: Card[][];
    roundScores: [number, number];
    scores: [number, number];
    currentBid: number;
    currentHighBidder: number;
    dealer: number;
    currentPlayer: number;
    trumpSuit: Suit | null;
    phase: Phase;
    tricks: TrickEntry[][];
    playedCards: Card[];
    currentTrick: TrickEntry[];
    trickWinner: number | null;
    playerCardsTaken: number[];
    numberOfRoundsPlayed: number;
    playingIterator: number;
    gameOver: boolean;
    winner: 0 | 1 | null;
    constructor(winThreshold?: number);
    reset(dealerOverride?: number): void;
    /** Start a new round (preserves scores, increments dealer) */
    newRound(): void;
    getActionMask(): number[];
    /**
     * Execute an action. Returns metadata about what happened for broadcasting.
     */
    step(action: number): {
        type: "bid" | "trumpChosen" | "cardPlayed" | "noValidPlay" | "trickResult" | "roundEnd" | "gameOver";
        data: Record<string, unknown>;
    }[];
    getHandData(seatIndex: number): CardData[];
    getPlayerCardCounts(): [number, number, number, number];
    private createDeck;
    private dealCards;
    private handleBid;
    private handleChooseSuit;
    private handlePlay;
    private discardAndFill;
    private noMoreValidPlaysAnyHand;
    private endRound;
    private resolveTrick;
    private checkGameEnd;
    isOffJack(card: Card): boolean;
    isValidPlay(card: Card): boolean;
}
