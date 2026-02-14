import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from pitch_env import PitchEnv
import sys
import time
import os

FileToInput = None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        FileToInput = sys.argv[1]
    else:
        print("No arguments provided (except script name).")

# MPS can be slower than CPU for small networks due to transfer overhead.
# Set USE_MPS=1 to try it, but CPU is the safe default for this model size.
if os.environ.get("USE_MPS") and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Shared feature layers: 2x expansion then compression
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # Dueling streams: separate "how good is this state" from
        # "how much better is each action than average"
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        features = self.features(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q = V(s) + (A(s,a) - mean(A(s,a)))
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

class Agent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.q_network = DQN(state_dim, action_dim).to(device)
        self.target_network = DQN(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)
        self.memory = deque(maxlen=100000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05

    def act(self, state, action_mask):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(np.where(action_mask == 1)[0])
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state).squeeze()
        # -inf for invalid actions so they're never chosen, even when
        # valid actions have negative Q-values
        q_values[torch.FloatTensor(action_mask).to(device) == 0] = float('-inf')
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        # Process entire batch at once instead of one sample at a time
        states = torch.FloatTensor(np.array([s for s, _, _, _, _ in minibatch])).to(device)
        actions = torch.LongTensor([a for _, a, _, _, _ in minibatch]).to(device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in minibatch]).to(device)
        next_states = torch.FloatTensor(np.array([ns for _, _, _, ns, _ in minibatch])).to(device)
        dones = torch.BoolTensor([d for _, _, _, _, d in minibatch]).to(device)

        # Q-values for the actions we actually took
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Best Q-values for next states from target network
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            next_q[dones] = 0.0
            target_q = rewards + self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def flatten_observation(obs, debug = False):
    flattened = []
    if (debug): print('keys: ', obs.keys())
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            if (debug): print('value array shape: ', value.shape)
            flattened.extend(value.flatten())
        elif isinstance(value, (int, np.integer)):
            if (debug): print('int array val', str(value))
            flattened.append(value)
    if (debug): print("Final flattened state size:", len(np.array(flattened)))
    return np.array(flattened)

def train_agents(FileToInput, num_episodes=200000):
    # Curriculum: start with short games, ramp to full 54-point games
    curriculum = [
        (0.00, 5),    # episodes 0-9%: first to 5
        (0.10, 10),   # episodes 10-29%: first to 10
        (0.30, 20),   # episodes 30-59%: first to 20
        (0.60, 35),   # episodes 60-79%: first to 35
        (0.80, 54),   # episodes 80-100%: full game
    ]

    env = PitchEnv()
    if (FileToInput is not None):
        try:
            with open(FileToInput, 'r', encoding='utf-8') as file:
                env.loadStateFromJsonString(file.read())
        except FileNotFoundError:
            print(f"Error: The file '{FileToInput}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
    state_dim = len(flatten_observation(env.reset()[0]))
    action_dim = env.action_space.n
    agents = [Agent(state_dim, action_dim) for _ in range(4)]

    target_update_freq = 500
    replay_freq = 4  # replay every N steps instead of every step
    global_step = 0
    checkpoint_freq = 10000  # save every 10k episodes
    log_freq = 1000  # print stats every 1k episodes

    # Linear epsilon decay: explore for first 60% of training, then exploit
    explore_episodes = int(num_episodes * 0.6)

    start_time = time.time()
    recent_rewards = deque(maxlen=1000)

    for episode in range(num_episodes):
        # Linear epsilon decay across all agents
        if episode < explore_episodes:
            eps = 1.0 - (1.0 - agents[0].epsilon_min) * (episode / explore_episodes)
        else:
            eps = agents[0].epsilon_min
        for agent in agents:
            agent.epsilon = eps

        # Set win threshold based on curriculum
        progress = episode / num_episodes
        threshold = curriculum[-1][1]
        for start_frac, thresh in curriculum:
            if progress >= start_frac:
                threshold = thresh
        env.win_threshold = threshold

        obs, _ = env.reset()
        done = False
        total_reward = [0, 0, 0, 0]

        while not done:
            current_player = env.current_player
            state = flatten_observation(obs)
            action = agents[current_player].act(state, obs['action_mask'])
            next_obs, reward, done, _, _ = env.step(action, obs)
            next_state = flatten_observation(next_obs)
            agents[current_player].remember(state, action, reward, next_state, done)
            if global_step % replay_freq == 0:
                agents[current_player].replay()
            obs = next_obs
            total_reward[current_player] += reward
            global_step += 1

            # Step-based target updates so learning stays stable during long games
            if global_step % target_update_freq == 0:
                for agent in agents:
                    agent.update_target_network()

        recent_rewards.append(sum(total_reward))

        if episode % log_freq == 0 and episode > 0:
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed
            remaining = (num_episodes - episode) / eps_per_sec
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Threshold: {threshold} | "
                f"Eps: {eps:.3f} | "
                f"Avg Reward (1k): {avg_reward:.1f} | "
                f"Steps: {global_step} | "
                f"{eps_per_sec:.1f} ep/s | "
                f"ETA: {remaining/3600:.1f}h"
            )

        if episode % checkpoint_freq == 0 and episode > 0:
            os.makedirs("checkpoints", exist_ok=True)
            for i, agent in enumerate(agents):
                torch.save(agent.q_network.state_dict(), f"checkpoints/agent_{i}_ep{episode}.pt")
            print(f"  Checkpoint saved at episode {episode}")

    return agents

# Train the agents
trained_agents = train_agents(FileToInput)

# Save final checkpoint
os.makedirs("checkpoints", exist_ok=True)
for i, agent in enumerate(trained_agents):
    torch.save(agent.q_network.state_dict(), f"checkpoints/agent_{i}_final.pt")

# Export to ONNX for the Node.js server
state_dim = len(flatten_observation(PitchEnv().reset()[0]))
dummy_input = torch.zeros(1, state_dim, device=device)
for i, agent in enumerate(trained_agents):
    agent.q_network.eval()
    torch.onnx.export(
        agent.q_network,
        dummy_input,
        f"agent_{i}_longtraining.onnx",
        input_names=["state"],
        output_names=["q_values"],
    )
print(f"Exported {len(trained_agents)} agents to ONNX")