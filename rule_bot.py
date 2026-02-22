"""
Deterministic heuristic bot for Pitch.

Used as a training opponent (50% of episodes) and evaluation baseline.
Public API: pick_action(env: PitchEnv) -> int
"""
from pitch_env import PitchEnv, Phase, Suit


def _card_points(card) -> int:
    """Points value of a card (3 for three, 1 for Ace/Jack/Ten/Joker/Two, else 0)."""
    if card.rank == 3:
        return 3
    if card.rank in {15, 12, 10, 11, 2}:
        return 1
    return 0


def _trump_score_for_suit(hand, suit: Suit, env: PitchEnv) -> int:
    """Compute the trump point value if `suit` were chosen as trump."""
    total = 0
    # Check for off-jack suit
    off_jack_suit = {
        Suit.CLUBS: Suit.SPADES,
        Suit.SPADES: Suit.CLUBS,
        Suit.DIAMONDS: Suit.HEARTS,
        Suit.HEARTS: Suit.DIAMONDS,
    }.get(suit)
    for card in hand:
        is_trump = card.suit == suit
        is_joker = card.rank == 11
        is_off_jack = (card.rank == 12 and card.suit == off_jack_suit)
        if is_trump or is_joker or is_off_jack:
            total += _card_points(card)
    return total


def _is_trump(card, env: PitchEnv) -> bool:
    return env._is_valid_play(card)


def _current_trick_winner_player(env: PitchEnv) -> int:
    """Return the player index currently winning the trick, or -1 if trick empty."""
    if not env.current_trick:
        return -1
    winner_tuple = max(env.current_trick,
                       key=lambda t: (env._is_valid_play(t[0]), t[0].rank))
    return winner_tuple[1]


def _current_trick_winner_rank(env: PitchEnv) -> int:
    """Return the rank of the currently winning card in the trick."""
    if not env.current_trick:
        return -1
    winner_tuple = max(env.current_trick,
                       key=lambda t: (env._is_valid_play(t[0]), t[0].rank))
    return winner_tuple[0].rank


def pick_action(env: PitchEnv) -> int:
    """Pick a legal action for env.current_player using deterministic heuristics."""
    mask = env._get_action_mask()
    valid = [i for i in range(len(mask)) if mask[i] == 1]
    cp = env.current_player
    hand = env.hands[cp]

    # --- BIDDING ---
    if env.phase == Phase.BIDDING:
        # Score each suit by trump points in hand
        best_score = 0
        for suit in Suit:
            score = _trump_score_for_suit(hand, suit, env)
            if score > best_score:
                best_score = score

        # Dealer forced bid if no one has bid yet
        if env.current_player == env.dealer and env.current_bid == 0:
            # Forced minimum bid of 5 (action 11)
            if 11 in valid:
                return 11
            # If 11 not in mask (shouldn't happen), fall through

        if best_score >= 5 and best_score > env.current_bid:
            # action = score + 6 (bid 5→action 11, ..., bid 10→action 16)
            action = min(best_score + 6, 16)
            if action in valid:
                return action
            # Try lower bids if exact not available
            for a in range(action, 10, -1):
                if a in valid:
                    return a

        # Pass
        if 10 in valid:
            return 10

    # --- CHOOSE SUIT ---
    elif env.phase == Phase.CHOOSESUIT:
        best_score = -1
        best_suit_action = 19  # Hearts default
        for suit in Suit:
            score = _trump_score_for_suit(hand, suit, env)
            if score > best_score:
                best_score = score
                best_suit_action = 19 + suit.value
        if best_suit_action in valid:
            return best_suit_action
        # Fallback to first valid suit action
        for a in range(19, 23):
            if a in valid:
                return a

    # --- PLAYING ---
    elif env.phase == Phase.PLAYING:
        # Get valid trump cards in hand (indices 0-9)
        play_valid = [i for i in valid if i <= 9]
        if not play_valid and 23 in valid:
            return 23

        trump_plays = [(i, hand[i]) for i in play_valid if _is_trump(hand[i], env)]
        non_trump_plays = [(i, hand[i]) for i in play_valid if not _is_trump(hand[i], env)]

        leading = len(env.current_trick) == 0

        if leading:
            # Lead highest trump; if no trump, highest overall
            if trump_plays:
                return max(trump_plays, key=lambda x: x[1].rank)[0]
            # No trump: play highest card
            if play_valid:
                return max(play_valid, key=lambda i: hand[i].rank)

        else:
            # Following
            winner_player = _current_trick_winner_player(env)
            winner_rank = _current_trick_winner_rank(env)
            partner_winning = (winner_player != -1 and winner_player % 2 == cp % 2)

            if partner_winning:
                # Partner is winning: throw lowest non-scoring valid card
                non_scorers = [(i, c) for i, c in
                               [(i, hand[i]) for i in play_valid]
                               if _card_points(c) == 0]
                if non_scorers:
                    return min(non_scorers, key=lambda x: x[1].rank)[0]
                # All valid cards score — throw lowest
                return min(play_valid, key=lambda i: hand[i].rank)

            else:
                # Opponent winning: try to beat with lowest winning trump
                # A card beats the winner if it's trump (valid play) and rank > winner_rank
                beating_trump = [(i, c) for i, c in trump_plays
                                 if c.rank > winner_rank]
                if beating_trump:
                    # Play lowest trump that beats current winner
                    return min(beating_trump, key=lambda x: x[1].rank)[0]
                # Can't beat: throw lowest non-scoring card
                all_cards = [(i, hand[i]) for i in play_valid]
                non_scorers = [(i, c) for i, c in all_cards if _card_points(c) == 0]
                if non_scorers:
                    return min(non_scorers, key=lambda x: x[1].rank)[0]
                # Everything scores: throw lowest
                return min(play_valid, key=lambda i: hand[i].rank)

    # Fallback: first valid action
    return valid[0]
