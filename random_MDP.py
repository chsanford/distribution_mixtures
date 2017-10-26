import random
import itertools
import numpy as np

class RandomMDP:
    """
    Class for generating a randomized MDP as described in section 6 of [Jiang et al., 2015],
    https://web.eecs.umich.edu/~baveja/Papers/gamma-AAMAS-final.pdf
    """

    def __init__(self, num_states, num_actions, gamma=0.9, seed=None):
        # Seed the random number generator for multiple experiments.
        self.ran = random.Random()
        self.ran.seed(seed)
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.transitions = self.generate_new_transitions()
        self.rewards = self.generate_new_reward()
        self.v_star, self.p_opt = self.solve()

    def generate_new_transitions(self):
        transitions = []
        # For each state,
        for state in range(self.num_states):
            transitions.append([])
            # For each action,
            for action in range(self.num_actions):
                # Generate a distribution over next states by,
                next_state_dist = [0.0] * self.num_states
                # Randomly select half of the states to transition to and assign them a random value in [0, 1).
                for each in self.ran.sample(xrange(self.num_states), self.num_states / 2):
                    next_state_dist[each] = self.ran.random()
                # Normalize.
                s = sum(next_state_dist)
                next_state_dist = [w/s for w in next_state_dist]
                transitions[state].append(next_state_dist)
        return np.array(transitions)

    def generate_new_reward(self):
        # Sample rewards uniformly and independently from [0,1) with additive Gaussian noise, s.d. = 0.1.
        return np.random.uniform(size=self.num_states) + np.random.normal(scale=0.1, size=self.num_states)

    def get_next_state(self, state, action):
        # Calculates a next state, given current state and action, according to transition matrix.
        next_state = -1
        s = 0
        threshold = self.ran.random()
        while s < threshold:
            next_state += 1
            s += self.transitions[state][action][next_state]
        return next_state

    def get_reward(self, state):
        # Returns the reward for a given state.
        return self.rewards[state]

    def solve(self, theta=1E-4, max_iter=-1):
        # Uses value iteration with threshold theta and/or maximum iterations max_iter to solve an MDP.
        diffs = np.full(self.num_states, float('inf'))
        v_curr, v_next = np.zeros(self.num_states), np.zeros(self.num_states)
        i = 0
        while (not all(diff <= theta for diff in diffs)) or (i < max_iter):
            v_next, _ = self.vi_update(v_curr)
            diffs = v_next - v_curr
            v_curr = v_next
            i += 1
        return self.vi_update(v_curr)

    def vi_update(self, v_curr):
        # A single update of value iteration.
        v_next = np.zeros(self.num_states)
        policy_curr = np.full(self.num_states, -1)
        for state in range(self.num_states):
            max_v = float('-inf')
            for action in range(self.num_actions):
                v = np.dot(self.rewards + np.array([self.gamma * v_curr[s] for s in range(self.num_states)]),
                           self.transitions[state][action])
                if v >= max_v:
                    max_v = v
                    policy_curr[state] = action
            v_next[state] = max_v
        return v_next, policy_curr

    def __str__(self):
        # Format string to print T and R matrices.
        states = ["s%d" % state for state in range(self.num_states)]
        actions = ["a%d" % action for action in range(self.num_actions)]
        rows = [row[0]+row[1] for row in itertools.product(states,actions)]
        row_format = "{:>20}" * (len(states) + 1)
        out = "Transitions:\n" + row_format.format("", *states) + '\n\n'
        for state, row in zip(rows, sum(np.ndarray.tolist(self.transitions),[])):
            out += row_format.format(state, *row) + '\n\n'
        out += "\n\nRewards:\n" + row_format.format("", *states) + '\n\n'
        out += row_format.format("", *self.rewards) + '\n\n'
        return out
