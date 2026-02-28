import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
def run(num_episodes: int = 12000,render=False):
    env = gym.make("FrozenLake-v1", render_mode="human" if render else None,is_slippery=False,map_name="8x8")
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    alpha = 0.1
    gamma = 0.99 
    epsilon = 1
    epsilo_min = 0.01
    epsilon_decay = 0.9995
    rnd = np.random.default_rng()
    reward_in_episode = np.zeros(num_episodes)
    for i in tqdm.tqdm(range(num_episodes)):
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
        epsilon = max(epsilo_min, epsilon * epsilon_decay)  # Decay epsilon , exponentially


        if epsilon == 0:
            alpha = 0.0001 # Reduce learning rate after exploration is done
        if reward == 1:
            reward_in_episode[i] = 1
    env.close()

    sum_reward = np.zeros(num_episodes)
    for i in range(num_episodes):
        sum_reward[i] = np.sum(reward_in_episode[max(0, i-100):i+1])
    plt.plot(sum_reward)
    plt.savefig("results/frozen_lake_q_learning_8.png")
    f=open("results/frozen_lake_q_table_8.pkl", "wb")
    pickle.dump(q_table, f)
    f.close()   


if __name__ == "__main__":
    run(num_episodes=25000)


