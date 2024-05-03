import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
from q_learning import get_win_rate as get_win_rate_q
from value_iteration import get_win_rate as get_win_rate_v

random_map = generate_random_map(size=4, p=0.75)

number_episodes = 1000
win_rate_q = get_win_rate_q(random_map, number_episodes)
win_rate_v = get_win_rate_v(random_map, number_episodes)

accuracies = [win_rate_q/number_episodes, win_rate_v/number_episodes]

plt.bar(['Q-learning', 'Value Iteration'], accuracies)
plt.xlabel('Method')
plt.ylabel('Accuracy %')
plt.title('Comparison of Q-learning and Value Iteration')
plt.show()

