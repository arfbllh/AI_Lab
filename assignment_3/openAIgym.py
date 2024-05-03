import gym
from gym.envs.toy_text.frozen_lake import generate_random_map


random_map = generate_random_map(size=8, p=0.6)

print(random_map)

env = gym.make('FrozenLake-v1', desc=random_map)

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)  
    print(f'state: {observation}, reward = {reward} cell = {random_map[observation]} terminated = {terminated} truncated = {truncated} {info}')
    if terminated or truncated:
        observation, info = env.reset()  

env.close()  