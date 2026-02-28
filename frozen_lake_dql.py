import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm

class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        self.fc1 = nn.Linear(in_states, h1_nodes) # Input layer to hidden layer
        self.fc2 = nn.Linear(h1_nodes, out_actions) # Hidden layer to output layer
    
    def forward(self, x):
        x = F.relu(self.fc1(x)) # Activation function for hidden layer
        x = self.fc2(x) # Output layer (Q-values for each action)
        return x

class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class FrozenLakeDQL():
    def __init__(self, env, memory_size=2000, batch_size=64, gamma=0.99, lr=0.001,network_sync_freq=10):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.network_sync_freq = network_sync_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = DQN(env.observation_space.n, 128, env.action_space.n).to(self.device)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.loff_fn = nn.MSELoss()
        self.optimizer = None # inside train function to avoid error when model is not yet initialized

        self.ACTIONS = ['L','D','R','U'] 

    def train(self, num_episodes=1000,is_slippery=False,render=False):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        env = gym.make("FrozenLake-v1", render_mode="human" if render else None,is_slippery=is_slippery,map_name="4x4")

        n_actions = env.action_space.n
        n_states = env.observation_space.n

        epsilon = 1.0
        memory = ReplayMemory(self.memory_size)

        policy_dqn = DQN(n_states, n_states, n_actions).to(self.device)
        target_dqn = DQN(n_states, n_states, n_actions).to(self.device)

        target_dqn.load_state_dict(policy_dqn.state_dict()) # Initialize target network with same weights as policy network
        
        print('Policy before training:')
        self.print_dqn(policy_dqn)

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.lr)
        # self.device = device  # Store device for use in other methods
        
        reward_in_episode = np.zeros(num_episodes)

        epsilon_hist = []

        step_count = 0

        for episode in tqdm.tqdm(range(num_episodes)):

            state = env.reset()[0]
            terminated , truncated = False, False

            while not (terminated or truncated):
                if random.random() < epsilon:
                    action = env.action_space.sample() # Explore: random action
                else:
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, n_states)).argmax().item() 

                next_state, reward, terminated, truncated, info = env.step(action)

                memory.append((state, action, reward, next_state, terminated)) # Store transition in replay memory

                state = next_state
                step_count += 1
            if not terminated:
                reward-=0.01 # Penalize for not reaching the goal to encourage faster learning
            if reward==1:
                reward_in_episode[episode] = 1
            

                if len(memory) >= self.batch_size and np.sum(reward_in_episode) > 0: # Start training only after we have enough samples and at least one reward
                    transitions = memory.sample(self.batch_size) # Sample a batch of transitions
                    self.optimize_model(policy_dqn, target_dqn, transitions) # Optimize the model using the sampled transitions

                    epsilon = max(epsilon - 1/episode, 0) # Decay epsilon linearly
                    epsilon_hist.append(epsilon)

                    if step_count>10:
                        step_count = 0
                        target_dqn.load_state_dict(policy_dqn.state_dict()) # Periodically update target network to match policy network

        env.close()

        torch.save(policy_dqn.state_dict(), "results/frozen_lake_dqn_4x4.pt")

        plt.figure(1)

        sum_rewards = np.zeros(num_episodes)
        for i in range(num_episodes):
            sum_rewards[i] = np.sum(reward_in_episode[max(0, i-100):i+1])
        plt.subplot(1,2,1)
        plt.plot(sum_rewards)
        plt.title("Rewards (moving sum over 100 episodes)")
        plt.xlabel("Episode")
        plt.ylabel("Sum of rewards in last 100 episodes")

        plt.subplot(1,2,2)
        plt.plot(epsilon_hist)
        plt.title("Epsilon Decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.savefig('frozen_lake_dql.png')

        self.print_dqn(policy_dqn)
    
    def test(self, num_episodes=5, render=True):
        env = gym.make("FrozenLake-v1", render_mode="human" if render else None,is_slippery=False,map_name="4x4")
        n_actions = env.action_space.n
        n_states = env.observation_space.n

        policy_dqn = DQN(n_states, n_states, n_actions)
        policy_dqn.load_state_dict(torch.load("results/frozen_lake_dqn_4x4.pt"))

        total_rewards = 0

        for episode in range(num_episodes):
            state = env.reset()[0]
            terminated , truncated = False, False

            while not (terminated or truncated):
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, n_states)).argmax().item() 
                state, reward, terminated, truncated, info = env.step(action)
                total_rewards += reward

        env.close()
        print(f'Average reward over {num_episodes} test episodes: {total_rewards/num_episodes:.2f}')

    def optimize_model(self, policy_dqn, target_dqn,transitions):
        ### optimzer is initialized in train function to avoid error when model is not yet initialized
        n_states = policy_dqn.fc1.in_features

        policy_q_list= []
        target_q_list = []

        for state, action, reward, next_state, terminated in transitions:
             
            if terminated:
                target = torch.FloatTensor([reward]).to(self.device)
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(reward + self.gamma * torch.max(target_dqn(self.state_to_dqn_input(next_state, n_states)))).to(self.device)
            
            curr_q = policy_dqn(self.state_to_dqn_input(state, n_states))
            policy_q_list.append(curr_q)

            target_q = target_dqn(self.state_to_dqn_input(state, n_states))
            target_q[action] = target
            target_q_list.append(target_q)
        
        loss = self.loff_fn(torch.stack(policy_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        ### Not able to see backward function in class initialization of mse error
        loss.backward()
        self.optimizer.step()

    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        if hasattr(self, 'device'):
            input_tensor = input_tensor.to(self.device)
        return input_tensor
    
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            if (s+1)%4==0:
                print() # Print a newline every 4 states


if __name__ == "__main__":
    agent = FrozenLakeDQL(None)
    agent.train(num_episodes=50000,is_slippery=False,render=False)    
    agent.test(num_episodes=5, render=True)                                 