"""Generate golden observation vectors from the Python engine for TS parity tests.

Run from the ML-Pitch-Theory directory with the venv active:
    source pitchenv/bin/activate
    python webserver/__tests__/fixtures/gen_parity_vectors.py

Writes parity_vectors.json next to this script. The TS test
(observationParity.test.ts) reconstructs each engine state and asserts that
flattenObservation() reproduces `expected` exactly. This locks Python<->TS
observation parity, including the on-jack/off-jack trick-winner tiebreak in
derived features 4 and 5.
"""
import json
import os
import sys

# Make the project root (ML-Pitch-Theory/) importable regardless of cwd.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np

from pitch_env import PitchEnv, Suit, Phase, Card
from train import flatten_observation
from rule_bot import pick_action as rule_bot_pick


def card_json(card):
    return {"s": (card.suit.value if card.suit is not None else None), "r": int(card.rank)}


def record(env):
    obs = env._get_observation()
    return {
        "seat": int(env.current_player),
        "phase": env.phase.value,
        "trumpSuit": (env.trump_suit.value if env.trump_suit is not None else None),
        "currentBid": int(env.current_bid),
        "currentHighBidder": int(env.current_high_bidder),
        "dealer": int(env.dealer),
        "currentPlayer": int(env.current_player),
        "numberOfRoundsPlayed": int(env.number_of_rounds_played),
        "scores": [int(x) for x in env.scores],
        "roundScores": [int(x) for x in env.round_scores],
        "playerCardsTaken": [int(x) for x in env.player_cards_taken],
        "hands": [[card_json(c) for c in env.hands[p]] for p in range(4)],
        "playedCards": [card_json(c) for c in env.played_cards],
        "currentTrick": [
            {"s": (t[0].suit.value if t[0].suit is not None else None),
             "r": int(t[0].rank), "seat": int(t[1])}
            for t in env.current_trick
        ],
        "expected": [round(float(x), 6) for x in flatten_observation(obs)],
        "ruleBotAction": int(rule_bot_pick(env)),
    }


def random_games(n_games=30, max_steps=400, cap=1500):
    records = []
    rng = np.random.default_rng(12345)
    for g in range(n_games):
        env = PitchEnv(win_threshold=54)
        env.reset(seed=1000 + g)
        steps = 0
        done = False
        while not done and steps < max_steps:
            # Sample more heavily in PLAYING (where derived features are active).
            if env.phase == Phase.PLAYING or steps % 4 == 0:
                records.append(record(env))
            mask = env._get_action_mask()
            valid = np.where(mask == 1)[0]
            action = int(rng.choice(valid))
            _, _, done, _, _ = env.step(action)
            steps += 1
    return records[:cap]


def crafted_offjack_partner_winning():
    """Off-jack (opponent) and on-jack (partner) both in the trick.
    Winner must be the on-jack → partner_winning (feat 5) = 1."""
    env = PitchEnv(win_threshold=54)
    env.reset(seed=0)
    env.phase = Phase.PLAYING
    env.trump_suit = Suit.CLUBS
    env.current_bid = 5
    env.current_high_bidder = 2
    env.dealer = 1
    env.current_player = 0
    env.playing_iterator = 2
    env.scores = [12, 9]
    env.round_scores = [3, 2]
    env.player_cards_taken = [0, 0, 0, 0]
    env.number_of_rounds_played = 1
    env.hands = [
        [Card(Suit.CLUBS, 15), Card(Suit.CLUBS, 5), Card(Suit.SPADES, 7)],
        [Card(Suit.CLUBS, 14), Card(Suit.HEARTS, 9)],
        [Card(Suit.CLUBS, 13), Card(Suit.DIAMONDS, 8)],
        [Card(Suit.CLUBS, 10), Card(Suit.HEARTS, 3)],
    ]
    # off-jack J♠ by seat 1 (opponent), on-jack J♣ by seat 2 (partner of seat 0)
    env.current_trick = [(Card(Suit.SPADES, 12), 1), (Card(Suit.CLUBS, 12), 2)]
    env.played_cards = [Card(Suit.SPADES, 12), Card(Suit.CLUBS, 12)]
    return record(env)


def crafted_onjack_beats_offjack():
    """Off-jack currently winning; root holds the on-jack → can_win (feat 4) = 1
    even though ranks are equal (both 12)."""
    env = PitchEnv(win_threshold=54)
    env.reset(seed=0)
    env.phase = Phase.PLAYING
    env.trump_suit = Suit.CLUBS
    env.current_bid = 4
    env.current_high_bidder = 3
    env.dealer = 2
    env.current_player = 0
    env.playing_iterator = 1
    env.scores = [5, 7]
    env.round_scores = [1, 0]
    env.player_cards_taken = [0, 0, 0, 0]
    env.number_of_rounds_played = 2
    env.hands = [
        [Card(Suit.CLUBS, 12), Card(Suit.HEARTS, 4)],  # on-jack J♣ + non-trump
        [Card(Suit.CLUBS, 9), Card(Suit.DIAMONDS, 5)],
        [Card(Suit.CLUBS, 8), Card(Suit.HEARTS, 6)],
        [Card(Suit.CLUBS, 7), Card(Suit.SPADES, 10)],
    ]
    # off-jack J♠ led by seat 3 (opponent); seat 0 to follow
    env.current_trick = [(Card(Suit.SPADES, 12), 3)]
    env.played_cards = [Card(Suit.SPADES, 12)]
    return record(env)


def main():
    records = random_games()
    records.append(crafted_offjack_partner_winning())
    records.append(crafted_onjack_beats_offjack())
    out_path = os.path.join(os.path.dirname(__file__), "parity_vectors.json")
    with open(out_path, "w") as f:
        json.dump(records, f)
    print(f"Wrote {len(records)} parity records to {out_path}")


if __name__ == "__main__":
    main()
