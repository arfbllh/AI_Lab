import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
import numpy as np
import time

def value_iteration(env, gamma = 1.0, max_iterations = 1000, delta = 1e-20):
    value_table = np.zeros(env.observation_space.n)
    delta_list = []

    for _ in range(max_iterations):
        updated_value_table = np.copy(value_table)
        max_change = 0
        for state in range(env.observation_space.n):
            Q_values = [sum([prob*(reward + gamma*updated_value_table[next_state]) 
                             for prob, next_state, reward, _ in env.P[state][action]]) 
                        for action in range(env.action_space.n)]
            best_action = max(Q_values)
            max_change  = max(max_change, np.abs(value_table[state] - best_action))
            value_table[state] = best_action
        delta_list.append(max_change)
        if max_change < delta:
            print(f'Converged at iteration {_:,}')
            break
    return value_table, delta_list
def find_policy(env, value_table, gamma = 1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_values = [sum([prob*(reward + gamma*value_table[next_state]) 
                         for prob, next_state, reward, _ in env.P[state][action]]) 
                    for action in range(env.action_space.n)]
        policy[state] = np.argmax(np.array(Q_values))
    return policy

def policy_simulation(policy, random_map, num_episodes = 1000, render = False, delay = 0.05):
    if render:
        env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=True, render_mode='human')
    else:
        env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=True)
    rewards = []
    wins = 0
    for episode in range(num_episodes):
        env.reset()
        state = 0
        episode_reward = 0
        done = False
        while not done:
                
            action = policy[state]
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                done = True
        rewards.append(episode_reward)
        if episode_reward > 0:
            wins += 1
    env.close()
    return wins


def get_win_rate(random_map, num_episode = 1000):
    env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=True)
    value_table, _ = value_iteration(env)
    policy = find_policy(env, value_table)
    wins = policy_simulation(policy.astype(int),random_map, num_episodes = num_episode)
    return wins

if __name__ == '__main__':
    random_map = generate_random_map(size=4, p=0.6)
    env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=True)
    value_table, delta_list = value_iteration(env)
    policy = find_policy(env, value_table)
    policy_simulation(policy.astype(int), random_map, num_episodes = 10000, render = False, delay = 0.2)
    plt.figure(figsize=(8, 4))
    plt.plot(delta_list, label='Max change in Value Function')
    plt.xlabel('Iternations')
    plt.ylabel('Max Change')
    plt.title("Convergence of Value Function")
    plt.legend()
    plt.grid(True)
    plt.show()

