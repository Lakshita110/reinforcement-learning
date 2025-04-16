import numpy as np
from collections import defaultdict
import gymnasium as gym


class QLearningAgent:

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):

        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def sample_greedy(self, state):
        return np.random.choice(np.flatnonzero(self.Q[state] == self.Q[state].max()))

    def sample_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.nA)
        else:
            return self.sample_greedy(state)

    def decay_epsilon(self, episode):
        new_epsilon = self.initial_epsilon * (self.epsilon_decay ** episode)
        self.epsilon = max(self.final_epsilon, new_epsilon)

    def train(self, num_episodes):
        episode_lengths = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_length = 0

            while not done:
                action = self.sample_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                # Update Q-value using the Bellman equation
                best_next_action = np.argmax(self.q_table[next_state])
                td_target = reward + self.discount_factor * \
                    self.q_table[next_state][best_next_action]
                td_delta = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.lr * td_delta

                state = next_state
                episode_length += 1

            episode_lengths.append(episode_length)
            self.decay_epsilon(episode)

        return episode_lengths
