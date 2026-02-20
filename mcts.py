"""Information Set Monte Carlo Tree Search (IS-MCTS) for Pitch.

Root-parallel IS-MCTS with batched GPU inference. Each search spawns N
independent trees (one per determinization), runs S simulation steps per
tree, then aggregates action votes across all trees.

Only searches during the PLAYING phase. Bidding and suit choice use the
DQN directly.
"""

import math

import numpy as np
import torch

from pitch_env import PitchEnv, Phase
from train import flatten_observation


# ---------------------------------------------------------------------------
# Determinization
# ---------------------------------------------------------------------------

def determinize(env: PitchEnv, root_player: int,
                rng: np.random.Generator) -> PitchEnv:
    """Clone env and randomly re-deal unknown cards to other players.

    All cards the root player can't see (opponent hands + deck) are pooled
    and redistributed randomly, preserving hand sizes and deck size.
    """
    sim = env.deep_copy()
    sim.np_random = rng

    unknown = []
    hand_sizes = {}
    for p in range(4):
        if p == root_player:
            continue
        unknown.extend(sim.hands[p])
        hand_sizes[p] = len(sim.hands[p])
    unknown.extend(sim.deck)
    deck_size = len(sim.deck)
    rng.shuffle(unknown)

    idx = 0
    for p in range(4):
        if p == root_player:
            continue
        sim.hands[p] = unknown[idx:idx + hand_sizes[p]]
        idx += hand_sizes[p]
    sim.deck = unknown[idx:idx + deck_size]

    return sim


def _step_env(env, action, obs):
    """Step env and return (obs, terminated_or_left_playing)."""
    obs, _, term, _, _ = env.step(action, obs)
    # Treat round boundaries as terminal — MCTS only searches current round
    done = term or env.phase != Phase.PLAYING
    return obs, done


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ['action', 'parent', 'children', 'visits', 'value']

    def __init__(self, action=-1, parent=None):
        self.action = action
        self.parent = parent
        self.children: dict[int, 'MCTSNode'] = {}
        self.visits = 0
        self.value = 0.0

    def ucb1(self, c) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, c) -> 'MCTSNode':
        return max(self.children.values(), key=lambda n: n.ucb1(c))


# ---------------------------------------------------------------------------
# Batched Root-Parallel IS-MCTS
# ---------------------------------------------------------------------------

class BatchedISMCTS:
    """Root-parallel IS-MCTS with batched GPU inference."""

    def __init__(self, q_network, device, num_envs=64, num_steps=8, c=1.41):
        self.q_network = q_network
        self.device = device
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.c = c

    def search(self, env: PitchEnv, player: int) -> int:
        """Run root-parallel IS-MCTS. Returns best action for player."""
        was_training = self.q_network.training
        self.q_network.eval()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

        N = self.num_envs
        rng = np.random.default_rng()
        trees = [MCTSNode() for _ in range(N)]

        for _ in range(self.num_steps):
            # Re-determinize from root state each step
            step_envs = []
            step_obs = []
            for i in range(N):
                child_rng = np.random.default_rng(rng.integers(2**63))
                sim = determinize(env, player, child_rng)
                step_envs.append(sim)
                step_obs.append(sim._get_observation())
            step_done = [False] * N

            # Select: walk each tree to a leaf, alternating between
            # root-player tree moves and batched opponent advances.
            nodes = list(trees)
            for _ in range(7):  # max ~6 root-player decisions per round
                any_needs_advance = False
                for i in range(N):
                    if step_done[i]:
                        continue
                    node = nodes[i]
                    while node.children and not step_done[i]:
                        if step_envs[i].current_player == player:
                            node = node.best_child(self.c)
                            step_obs[i], step_done[i] = _step_env(
                                step_envs[i], node.action, step_obs[i])
                        else:
                            any_needs_advance = True
                            break
                    nodes[i] = node

                if not any_needs_advance:
                    break
                self._advance_opponents(
                    step_envs, step_obs, step_done, player)

            # Expand: add one child per tree at the leaf
            for i in range(N):
                if step_done[i]:
                    continue
                if step_envs[i].current_player != player:
                    # Opponent's turn at leaf — advance and evaluate
                    self._advance_opponents(
                        step_envs, step_obs, step_done, player)
                if step_done[i]:
                    continue
                if step_envs[i].current_player != player:
                    continue
                mask = step_obs[i]['action_mask']
                valid = [a for a in range(24) if mask[a]]
                unexpanded = [a for a in valid if a not in nodes[i].children]
                if unexpanded:
                    action = unexpanded[rng.integers(len(unexpanded))]
                    child = MCTSNode(action=action, parent=nodes[i])
                    nodes[i].children[action] = child
                    step_obs[i], step_done[i] = _step_env(
                        step_envs[i], action, step_obs[i])
                    nodes[i] = child

            # Advance opponents at leaves so NN evaluates from root player's
            # perspective (the Q-network was trained on agent-team observations).
            self._advance_opponents(
                step_envs, step_obs, step_done, player)

            # Evaluate leaves (batched)
            values = self._batch_evaluate(
                step_envs, step_obs, step_done, player)

            # Backup
            for i in range(N):
                node = nodes[i]
                while node is not None:
                    node.visits += 1
                    node.value += values[i]
                    node = node.parent

        # Aggregate action visits across all trees
        action_visits: dict[int, int] = {}
        for tree in trees:
            for action, child in tree.children.items():
                action_visits[action] = action_visits.get(action, 0) + child.visits

        if was_training:
            self.q_network.train()

        if not action_visits:
            return self._fallback_action(env, player)
        return max(action_visits, key=action_visits.get)

    def _advance_opponents(self, envs, obs_list, done, player):
        """Advance all envs through opponent moves using batched DQN."""
        for _ in range(7):  # 3 opponents + trick boundary + 3 more
            need = [i for i in range(len(envs))
                    if not done[i] and envs[i].current_player != player]
            if not need:
                break
            states = np.array([flatten_observation(obs_list[i]) for i in need])
            masks = np.array([obs_list[i]['action_mask'] for i in need])
            actions = self._batch_greedy(states, masks)
            for j, i in enumerate(need):
                obs_list[i], done[i] = _step_env(
                    envs[i], int(actions[j]), obs_list[i])

    def _batch_greedy(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Batched greedy action selection. Returns (B,) int array."""
        with torch.no_grad():
            t = torch.tensor(states, dtype=torch.float32, device=self.device)
            q = self.q_network(t).cpu().numpy()
        q[masks == 0] = -np.inf
        return np.argmax(q, axis=1)

    def _batch_evaluate(self, envs, obs_list, done, player) -> list:
        """Evaluate all leaf states in one batched forward pass."""
        team = player % 2
        values = [0.0] * len(envs)
        need_nn = []

        for i in range(len(envs)):
            if done[i]:
                s_ours, s_theirs = envs[i].scores[team], envs[i].scores[1 - team]
                values[i] = 1.0 if s_ours > s_theirs else (-1.0 if s_theirs > s_ours else 0.0)
            else:
                need_nn.append(i)

        if need_nn:
            states = np.array([flatten_observation(obs_list[i]) for i in need_nn])
            with torch.no_grad():
                t = torch.tensor(states, dtype=torch.float32, device=self.device)
                q = self.q_network(t)
                v = torch.tanh(q.max(dim=1).values).cpu().numpy()
            for j, i in enumerate(need_nn):
                values[i] = float(v[j])

        return values

    def _fallback_action(self, env, player) -> int:
        """Use DQN directly if search produced no children."""
        obs = env._get_observation()
        state = flatten_observation(obs)
        mask = obs['action_mask']
        with torch.no_grad():
            q = self.q_network(
                torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            ).squeeze(0).cpu().numpy()
        q[mask == 0] = -np.inf
        return int(np.argmax(q))
