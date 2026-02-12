export var Suit;
(function (Suit) {
    Suit[Suit["HEARTS"] = 0] = "HEARTS";
    Suit[Suit["DIAMONDS"] = 1] = "DIAMONDS";
    Suit[Suit["CLUBS"] = 2] = "CLUBS";
    Suit[Suit["SPADES"] = 3] = "SPADES";
})(Suit || (Suit = {}));
export var Phase;
(function (Phase) {
    Phase[Phase["BIDDING"] = 0] = "BIDDING";
    Phase[Phase["CHOOSESUIT"] = 1] = "CHOOSESUIT";
    Phase[Phase["PLAYING"] = 2] = "PLAYING";
})(Phase || (Phase = {}));
export function cardToData(card) {
    return { suit: card.suit !== null ? card.suit : null, rank: card.rank };
}
export function cardPoints(card) {
    if (card.rank === 15 || card.rank === 12 || card.rank === 10)
        return 1; // Ace, Jack, 10
    if (card.rank === 11)
        return 1; // Joker
    if (card.rank === 3)
        return 3;
    if (card.rank === 2)
        return 1;
    return 0;
}
export const RANK_NAMES = {
    11: "Joker",
    12: "J",
    13: "Q",
    14: "K",
    15: "A",
};
export const SUIT_NAMES = {
    [Suit.HEARTS]: "Hearts",
    [Suit.DIAMONDS]: "Diamonds",
    [Suit.CLUBS]: "Clubs",
    [Suit.SPADES]: "Spades",
};
// Off-jack pairs: same color, different suit
export const OFF_JACK_MAP = {
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
