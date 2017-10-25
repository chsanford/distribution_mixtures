import random
import itertools

class RandomMDP:
    """
    Class for generating a randomized MDP as described in section 6 of [Jiang et al., 2015],
    https://web.eecs.umich.edu/~baveja/Papers/gamma-AAMAS-final.pdf
    """

    ran = random.Random()

    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.transitions = []
        self.generate_new_transitions()
        self.rewards = []
        self.generate_new_reward()

    def generate_new_transitions(self):
        # For each state,
        for state in range(self.num_states):
            self.transitions.append([])
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
                self.transitions[state].append(next_state_dist)
        return

    def generate_new_reward(self):
        # Sample rewards uniformly and independently from [0,1) with additive Gaussian noise, s.d. = 0.1.
        self.rewards = [self.ran.random() + self.ran.gauss(0, 0.1) for state in range(self.num_states)]
        return

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

    def __str__(self):
        # Format string to print T and R matrices.
        states = ["s%d" % state for state in range(self.num_states)]
        actions = ["a%d" % action for action in range(self.num_actions)]
        rows = [row[0]+row[1] for row in itertools.product(states,actions)]
        row_format = "{:>20}" * (len(states) + 1)
        out = "Transitions:\n" + row_format.format("", *states) + '\n\n'
        for state, row in zip(rows, sum(self.transitions,[])):
            out += row_format.format(state, *row) + '\n\n'
        out += "\n\nRewards:\n" + row_format.format("", *states) + '\n\n'
        out += row_format.format("", *self.rewards) + '\n\n'
        return out
