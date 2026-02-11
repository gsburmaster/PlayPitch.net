import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from pitch_env import PitchEnv
import sys 

FileToInput = None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        FileToInput = sys.argv[1]
    else:
        print("No arguments provided (except script name).")

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


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
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

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

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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

def train_agents(FileToInput, num_episodes=10000):
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

    target_update_freq = 500  # update target network every N steps
    global_step = 0

    for episode in range(num_episodes):
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
            agents[current_player].replay()
            obs = next_obs
            total_reward[current_player] += reward
            global_step += 1

            # Step-based target updates so learning stays stable during long games
            if global_step % target_update_freq == 0:
                for agent in agents:
                    agent.update_target_network()

        if episode % 100 == 0:
            print(f"Episode: {episode}, Threshold: {threshold}, Rewards: {total_reward}")

    return agents

# Train the agents
trained_agents = train_agents(FileToInput)

# Export to ONNX for the Node.js server
state_dim = len(flatten_observation(PitchEnv().reset()[0]))
dummy_input = torch.zeros(1, state_dim, device=device)
for i, agent in enumerate(trained_agents):
    agent.q_network.eval()
    torch.onnx.export(
        agent.q_network,
        dummy_input,
        f"agent_{i}.onnx",
        input_names=["state"],
        output_names=["q_values"],
    )
print(f"Exported {len(trained_agents)} agents to ONNX")