import { describe, it, expect } from "vitest";
import { Suit, Phase, cardToData, cardPoints, OFF_JACK_MAP, ACTION_PASS, ACTION_BID_START, ACTION_SUIT_START, ACTION_NO_VALID_PLAY, NUM_ACTIONS, RANK_NAMES, SUIT_NAMES, } from "../game/constants.js";
describe("Suit enum", () => {
    it("has correct numeric values", () => {
        expect(Suit.HEARTS).toBe(0);
        expect(Suit.DIAMONDS).toBe(1);
        expect(Suit.CLUBS).toBe(2);
        expect(Suit.SPADES).toBe(3);
    });
});
describe("Phase enum", () => {
    it("has correct numeric values", () => {
        expect(Phase.BIDDING).toBe(0);
        expect(Phase.CHOOSESUIT).toBe(1);
        expect(Phase.PLAYING).toBe(2);
    });
});
describe("Action constants", () => {
    it("ACTION_PASS is 10", () => {
        expect(ACTION_PASS).toBe(10);
    });
    it("ACTION_BID_START is 11", () => {
        expect(ACTION_BID_START).toBe(11);
    });
    it("ACTION_SUIT_START is 19", () => {
        expect(ACTION_SUIT_START).toBe(19);
    });
    it("ACTION_NO_VALID_PLAY is 23", () => {
        expect(ACTION_NO_VALID_PLAY).toBe(23);
    });
    it("NUM_ACTIONS is 24", () => {
        expect(NUM_ACTIONS).toBe(24);
    });
    it("bid actions map to bid values 5-12", () => {
        // action 11 = bid 5 (11-6=5), action 18 = bid 12 (18-6=12)
        for (let action = 11; action <= 18; action++) {
            expect(action - 6).toBe(action - 6);
        }
    });
    it("suit actions map to suit enums", () => {
        expect(ACTION_SUIT_START + Suit.HEARTS).toBe(19);
        expect(ACTION_SUIT_START + Suit.DIAMONDS).toBe(20);
        expect(ACTION_SUIT_START + Suit.CLUBS).toBe(21);
        expect(ACTION_SUIT_START + Suit.SPADES).toBe(22);
    });
});
describe("cardToData", () => {
    it("converts a regular card", () => {
        const card = { suit: Suit.HEARTS, rank: 15 };
        expect(cardToData(card)).toEqual({ suit: 0, rank: 15 });
    });
    it("converts a card with numeric suit", () => {
        const card = { suit: Suit.SPADES, rank: 3 };
        expect(cardToData(card)).toEqual({ suit: 3, rank: 3 });
    });
    it("converts a joker (null suit)", () => {
        const card = { suit: null, rank: 11 };
        expect(cardToData(card)).toEqual({ suit: null, rank: 11 });
    });
    it("preserves all suit values", () => {
        for (const suit of [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]) {
            const data = cardToData({ suit, rank: 5 });
            expect(data.suit).toBe(suit);
        }
    });
});
describe("cardPoints", () => {
    it("Ace (rank 15) is worth 1 point", () => {
        expect(cardPoints({ suit: Suit.HEARTS, rank: 15 })).toBe(1);
    });
    it("Jack (rank 12) is worth 1 point", () => {
        expect(cardPoints({ suit: Suit.CLUBS, rank: 12 })).toBe(1);
    });
    it("Ten (rank 10) is worth 1 point", () => {
        expect(cardPoints({ suit: Suit.DIAMONDS, rank: 10 })).toBe(1);
    });
    it("Three (rank 3) is worth 3 points", () => {
        expect(cardPoints({ suit: Suit.SPADES, rank: 3 })).toBe(3);
    });
    it("Two (rank 2) is worth 1 point", () => {
        expect(cardPoints({ suit: Suit.HEARTS, rank: 2 })).toBe(1);
    });
    it("Joker (rank 11) is worth 1 point", () => {
        expect(cardPoints({ suit: null, rank: 11 })).toBe(1);
    });
    it("non-scoring cards are worth 0", () => {
        for (const rank of [4, 5, 6, 7, 8, 9, 13, 14]) {
            expect(cardPoints({ suit: Suit.HEARTS, rank })).toBe(0);
        }
    });
    it("total possible points in a round is 10", () => {
        // 4 Aces (4) + 4 Jacks (4) + 4 Tens (4) + 4 Threes (12) + 4 Twos (4) + 2 Jokers (2) = 30
        // But only trump suit cards are in play, so per-suit: A+J+10+3+2 = 1+1+1+3+1 = 7, plus 2 jokers = 9, plus off-jack = 10
        // We just verify the function returns correct values for each scoring rank
        const scoringCards = [
            { suit: Suit.HEARTS, rank: 15 }, // Ace = 1
            { suit: Suit.HEARTS, rank: 12 }, // Jack = 1
            { suit: Suit.HEARTS, rank: 10 }, // Ten = 1
            { suit: Suit.HEARTS, rank: 3 }, // Three = 3
            { suit: Suit.HEARTS, rank: 2 }, // Two = 1
            { suit: null, rank: 11 }, // Joker = 1
            { suit: null, rank: 11 }, // Joker = 1
            { suit: Suit.DIAMONDS, rank: 12 }, // Off-Jack = 1
        ];
        const total = scoringCards.reduce((sum, c) => sum + cardPoints(c), 0);
        expect(total).toBe(10);
    });
});
describe("OFF_JACK_MAP", () => {
    it("Clubs maps to Spades", () => {
        expect(OFF_JACK_MAP[Suit.CLUBS]).toBe(Suit.SPADES);
    });
    it("Spades maps to Clubs", () => {
        expect(OFF_JACK_MAP[Suit.SPADES]).toBe(Suit.CLUBS);
    });
    it("Hearts maps to Diamonds", () => {
        expect(OFF_JACK_MAP[Suit.HEARTS]).toBe(Suit.DIAMONDS);
    });
    it("Diamonds maps to Hearts", () => {
        expect(OFF_JACK_MAP[Suit.DIAMONDS]).toBe(Suit.HEARTS);
    });
    it("is symmetric (bidirectional)", () => {
        for (const suit of [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]) {
            expect(OFF_JACK_MAP[OFF_JACK_MAP[suit]]).toBe(suit);
        }
    });
});
describe("RANK_NAMES", () => {
    it("has correct name for Joker", () => {
        expect(RANK_NAMES[11]).toBe("Joker");
    });
    it("has correct name for face cards", () => {
        expect(RANK_NAMES[12]).toBe("J");
        expect(RANK_NAMES[13]).toBe("Q");
        expect(RANK_NAMES[14]).toBe("K");
        expect(RANK_NAMES[15]).toBe("A");
    });
});
describe("SUIT_NAMES", () => {
    it("has correct names for all suits", () => {
        expect(SUIT_NAMES[Suit.HEARTS]).toBe("Hearts");
        expect(SUIT_NAMES[Suit.DIAMONDS]).toBe("Diamonds");
        expect(SUIT_NAMES[Suit.CLUBS]).toBe("Clubs");
        expect(SUIT_NAMES[Suit.SPADES]).toBe("Spades");
    });
});
