import sys
import gym
import numpy as np
from collections import defaultdict
from plot_utils import plot_blackjack_values, plot_policy

env = gym.make('Blackjack-v0')


def play_episode(env):
    episode = []
    state = env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def update_Q(episode, N, R, Q, gamma):
    G = 0
    for s, a, r in episode:
        first_occurence_idx = next(i for i, x in enumerate(episode) if x[0] == s)
        discounted_future_reward = sum([e[2] * (gamma ** i) for i, e in enumerate(episode[first_occurence_idx:])])
        N[s][a] += 1
        R[s][a] += discounted_future_reward
        Q[s][a] = R[s][a] / N[s][a]
    return 0


def mc_predict(env, num_episodes, gamma=0.1):
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    R = defaultdict(lambda: np.zeros(env.action_space.n))
    for i_episode in range(num_episodes):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
        episode = play_episode(env)
        update_Q(episode, N, R, Q, gamma)
    return Q


Q = mc_predict(env, 1000000)
V_to_plot = dict(
    (k, (k[0] > 18) * (np.dot([0.8, 0.2], v)) + (k[0] <= 18) * (np.dot([0.2, 0.8], v))) for k, v in Q.items())
plot_blackjack_values(V_to_plot)