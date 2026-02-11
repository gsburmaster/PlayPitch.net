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
        print(f'initial input dimention for layer 1: {input_dim}')
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters())
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def act(self, state, action_mask):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(np.where(action_mask == 1)[0])
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        valid_q_values = q_values.squeeze() * torch.FloatTensor(action_mask)
        return valid_q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def print_size(self):
        print(self.state_dim, 'self state dim')
        return
    def replay(self,env):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                print(next_state.size())
                
                try:
                    target = reward + self.gamma * self.target_network(next_state).max(1)[0].item()
                except:
                    #env.print_state()
                    print(state)                    
                target = reward + self.gamma * self.target_network(next_state).max(1)[0].item()
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.q_network(state)
            target_f[0][action] = target
            loss = nn.MSELoss()(self.q_network(state), target_f)
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

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = [0, 0, 0, 0]

        while not done:
            current_player = env.current_player
            state = flatten_observation(obs)
            action = agents[current_player].act(state, obs['action_mask'])
            next_obs, reward, done, _, _ = env.step(action,obs)
            next_state = flatten_observation(next_obs)
            if (len(next_state) != len(state)):
                print('\ncurrent step:')
                print(state)
                print('\nobs:')
                print(obs)
                print('\n\n\n\nNext step:')
                print(next_state)
                print('\nnextObs')
                print(next_obs)
            try:
                agents[current_player].remember(state, action, reward, next_state, done)
                agents[current_player].replay(env)
            except RuntimeError:
                env.print_state()
                env.saveStateToFileAsJson('jsonErr')
                raise Exception
            obs = next_obs
            total_reward[current_player] += reward
            

        # Update target networks
        for agent in agents:
            agent.update_target_network()

        if episode % 100 == 0:
            print(f"Episode: {episode}, Rewards: {total_reward}")

    return agents

# Train the agents
trained_agents = train_agents(FileToInput)

# You can now use these trained agents to play against each other or evaluate their performance
model_scripted = torch.jit.script(trained_agents[0]) # Export to TorchScript
model_scripted.save('model_scripted.pt') # Save