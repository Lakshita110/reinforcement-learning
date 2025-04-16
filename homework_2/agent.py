import numpy as np


def _initialize_Q(env):
    Q = {}
    for row in range(env.nS[0]):
        for col in range(env.nS[1]):
            for y_speed in range(-4, 1):
                for x_speed in range(-4, 5):
                    Q[(np.int64(row), np.int64(col), y_speed, x_speed)
                      ] = np.zeros(env.nA)
    return Q


class OnPolicyMonteCarloAgent:
    """
    On-policy first-visit Monte Carlo control agent.
    Uses epsilon-greedy policy which gradually becomes more greedy.
    """

    def __init__(self, env, gamma=0.9, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.01):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = _initialize_Q(env)
        self.returns = {(s, a): [] for s in self.Q for a in range(env.nA)}
        self.policy = {s: np.ones(env.nA) / env.nA for s in self.Q}

    def sample_greedy(self, state):
        return np.random.choice(np.flatnonzero(self.Q[state] == self.Q[state].max()))

    def sample_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.nA)
        else:
            return self.sample_greedy(state)

    def update_policy(self):
        for state in self.Q.keys():
            best_action = self.sample_greedy(state)
            self.policy[state] = np.ones(
                self.env.nA) * (self.epsilon / self.env.nA)
            self.policy[state][best_action] += (1 - self.epsilon)

    def train(self, episodes=1000):
        episode_lengths = []
        for i in range(episodes):
            episode = []
            state = self.env.reset()[0]
            state = (state[0], state[1], 0, 0)
            done = False

            while not done:
                action = self.sample_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state

            G = 0
            visited = set()
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward

                if (state, action) not in visited:
                    visited.add((state, action))
                    self.returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(
                        self.returns[(state, action)])

            self.update_policy()

            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay)

            if i % 100 == 0:
                print(
                    f'Episode {i} completed, length: {len(episode)}, epsilon: {self.epsilon:.4f}')

            episode_lengths.append(len(episode))

        print("Training completed")
        return episode_lengths


class OffPolicyMonteCarloAgent:
    """
    Off-policy Monte Carlo control agent using weighted importance sampling.
    Target policy is greedy, behavior policy is epsilon-greedy.
    """

    def __init__(self, env, gamma=0.9, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = _initialize_Q(env)
        self.C = {(s, a): 0 for s in self.Q for a in range(
            env.nA)}

    def sample_greedy(self, state):
        return np.random.choice(np.flatnonzero(self.Q[state] == self.Q[state].max()))

    def sample_behavior_policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.nA)
        else:
            return self.sample_greedy(state)

    def target_policy_probability(self, state, action):
        optimal_actions = np.flatnonzero(self.Q[state] == self.Q[state].max())
        if action in optimal_actions:
            return 1.0 / len(optimal_actions)
        return 0.0

    def behavior_policy_probability(self, state, action):
        optimal_actions = np.flatnonzero(self.Q[state] == self.Q[state].max())
        if action in optimal_actions:
            return self.epsilon / self.env.nA + (1 - self.epsilon) / len(optimal_actions)
        else:
            return self.epsilon / self.env.nA

    def train(self, episodes=1000):
        episode_lengths = []
        for i in range(episodes):
            episode = []
            state = self.env.reset()[0]
            state = (state[0], state[1], 0, 0)
            done = False

            while not done:
                action = self.sample_behavior_policy(
                    state)
                next_state, reward, done, _ = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state

            G = 0
            W = 1.0

            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward

                self.C[(state, action)] += W
                if self.C[(state, action)] > 0:
                    self.Q[state][action] += (W / self.C[(state, action)]
                                              ) * (G - self.Q[state][action])

                if self.target_policy_probability(state, action) == 0.0:
                    break

                W *= self.target_policy_probability(
                    state, action) / self.behavior_policy_probability(state, action)

            if i % 100 == 0:
                print(f'Episode {i} completed, length: {len(episode)}')

            episode_lengths.append(len(episode))

        print("Training completed")
        return episode_lengths


__all__ = ["OnPolicyMonteCarloAgent", "OffPolicyMonteCarloAgent"]
