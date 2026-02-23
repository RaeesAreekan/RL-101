import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
def run(num_episodes: int = 12000,render=False):
    env = gym.make("FrozenLake-v1", render_mode="human" if render else None,is_slippery=False,map_name="4x4")
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    alpha = 0.9
    gamma = 0.9 
    epsilon = 1
    epsilon_decay = 0.0001
    rnd = np.random.default_rng()
    reward_in_episode = np.zeros(num_episodes)
    for i in range(num_episodes):
        state = env.reset()[0]
        terminated , truncated = False, False
        while not (terminated or truncated):
            if rnd.random() < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                action = np.argmax(q_table[state,:])  # Greedy action
            next_state, reward, terminated, truncated, info = env.step(action)
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
            state = next_state
        epsilon = max(epsilon - epsilon_decay, 0)


        if epsilon == 0:
            alpha = 0.0001 # Reduce learning rate after exploration is done
        if reward == 1:
            reward_in_episode[i] = 1
    env.close()

    sum_reward = np.zeros(num_episodes)
    for i in range(num_episodes):
        sum_reward[i] = np.sum(reward_in_episode[max(0, i-100):i+1])
    plt.plot(sum_reward)
    plt.savefig("frozen_lake_q_learning.png")
    f=open("frozen_lake_q_table.pkl", "wb")
    pickle.dump(q_table, f)
    f.close()   


if __name__ == "__main__":
    run(num_episodes=15000)


