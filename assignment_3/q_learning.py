import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
import numpy as np
import time

def init_env(random_map):
    env = gym.make('FrozenLake-v1', desc = random_map, is_slippery=True)
    qtable = np.zeros((env.observation_space.n, env.action_space.n))
    return env, qtable

def train_agent(env, qtable, tot_episodes, max_steps, alpha, gamma, epsilon, max_epsilon, min_epsilon, decay_rate):
    rewards = []

    for episode in range(tot_episodes):
        env.reset()
        state = 0
        episode_reward = 0
        for _ in range(max_steps):
            exp_exp_tradeoff = np.random.uniform(0, 1)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[state, :])
            else:
                action = env.action_space.sample()
            new_state, reward, done, _, _ = env.step(action)
            qtable[state, action] = qtable[state, action] + alpha*(reward + gamma*np.max(qtable[new_state, :]) - qtable[state, action])
            state = new_state
            episode_reward += reward
            if done:
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        rewards.append(episode_reward)
    return qtable, rewards

def evaluate_agent(env, qtable, tot_episodes, tot_steps=96):
    wins = 0
    for episode in range(tot_episodes):
        state, step, terminated = 0, 0, False
        env.reset()
        while not terminated:
            action = np.argmax(qtable[state, :])
            new_state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                if new_state == 15:
                    print('Goal is reached episode:', episode)
                    wins += 1
                break
               
            if step > tot_steps:
                break
            state = new_state
            step += 1
    env.close()
    return wins
    

def get_win_rate(random_map, tot_episodes=1000):
    env, qtable = init_env(random_map)
    qtable, _ = train_agent(env, qtable, 20000, 96, 0.6, 0.9, 1.0, 1.0, 0.01, 0.001)
    wins = evaluate_agent(env, qtable, tot_episodes)
    return wins
    
if __name__ == '__main__':
    random_map = generate_random_map(size=4, p=0.6)
    env, qtable = init_env(random_map)
    total_episodes = 25000
    max_steps = 96
    alpha = 0.75
    gamma = 0.9
    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.001
    qtable, rewards = train_agent(env, qtable, total_episodes, max_steps, alpha, gamma, epsilon, max_epsilon, min_epsilon, decay_rate)

    evaluate_agent(env, qtable, 1000)

    cum_avg_rewards = np.cumsum(rewards)/np.arange(1, total_episodes+1)
    plt.figure(figsize=(8, 4))
    plt.plot(cum_avg_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Q-Learning Training')
    plt.legend(['Average Reward per epsoide'])
    plt.grid(True)
    plt.show()

