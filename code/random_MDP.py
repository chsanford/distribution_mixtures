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
        np.random.seed(seed)
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.transitions = self.generate_new_transitions()
        self.rewards = self.generate_new_reward()
        self.p_opt, self.v_star, self.q_star = self.solve()

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

    def solve(self, theta=1E-10, max_iter=-1):
        # Uses value iteration with threshold theta and/or maximum iterations max_iter to solve an MDP.
        diffs = np.full(self.num_states, float('inf'))
        v_curr, v_next = np.zeros(self.num_states), np.zeros(self.num_states)
        i = 0
        while (not all(diff <= theta for diff in diffs)) or (i < max_iter):
            v_next, _ = self.vi_update(v_curr)
            diffs = v_next - v_curr
            v_curr = v_next
            i += 1
        v_star, _ = self.vi_update(v_curr)
        q_star = np.dot(self.transitions, self.rewards) + np.dot(self.transitions, v_star) * self.gamma
        p_star = [[] for i in range(self.num_states)]
        for state, action in np.argwhere(q_star == np.transpose([np.amax(q_star, 1)])):
            p_star[state].append(action)
        p_opt = [""]
        for action_list in p_star:
            p_opt = [pol + str(action) for action in action_list for pol in p_opt]
        return p_opt, v_star, q_star

    def vi_update(self, v_curr):
        # A single update of value iteration.
        v_next = np.zeros(self.num_states)
        policy_curr = np.full(self.num_states, -1)
        for state in range(self.num_states):
            max_v = float('-inf')
            for action in range(self.num_actions):
                v = np.dot(self.transitions[state][action],
                           self.rewards + np.array([self.gamma * v_curr[s] for s in range(self.num_states)]))
                if v >= max_v:
                    max_v = v
                    policy_curr[state] = action
            v_next[state] = max_v
        return v_next, policy_curr

    def sample(self, sample_size, dist=None):
        sample = []
        sa = np.array([[(s, a) for a in range(self.num_actions)] for s in range(self.num_states)],
                      dtype=[('state','i4'), ('action','i4')])
        for _ in range(sample_size):
            if type(dist) != type(None):
                s, a = np.random.choice(sa.flatten(), p=np.asarray(dist).flatten())
            else:
                s, a = np.random.choice(sa.flatten())
            sp = self.get_next_state(s, a)
            r = self.get_reward(sp)
            sample.append((s, a, r, sp))
        return sample

    def state_action_dist(self, pol):
        state_dist = np.transpose(np.asarray([self.stationary_dist(pol)]))
        sa_dist = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            sa_dist[s][int(pol[s])] = 1.0
        return np.asarray(state_dist * sa_dist)


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
        out += "\n\nPolicies:\n"
        out += "\n".join(self.p_opt) + "\n\n"
        return out

    def stationary_dist(self, pol):
        pol = np.asarray([int(p) for p in pol]).reshape(-1)
        assert self.num_actions==2
        ### ADAPTED FROM CLAYTONS CODE. WORKS ONLY FOR 2 ACTION MDPS
        ## Set up transition probabilities for the policy
        success_prob = 0.9
        # Probability of choosing each action
        adj = (1 - success_prob) / (self.num_actions - 1)
        action_dist = np.transpose((np.eye(self.num_actions)*(success_prob-adj))[pol] + adj)

        # Probability of choosing each state from a prior state given the policy
        next_state_dist = []
        for state_index in range(self.num_states):
            next_state_dist.append(
                np.multiply(action_dist[0][state_index], self.transitions[state_index][0]) +
                np.multiply(action_dist[1][state_index], self.transitions[state_index][1]))
        next_state_dist = np.transpose(np.asarray(next_state_dist))

        # Matrix used to find probabilities of the stationary distribution
        stationary_matrix = next_state_dist - np.identity(self.num_states)
        stationary_matrix[self.num_states - 1] = np.asarray([1] * self.num_states)

        # Stationary distribution over states given policy
        return np.transpose(np.matmul(
            np.linalg.inv(stationary_matrix), np.asarray([[0]] * (self.num_states - 1) + [[1]]))).flatten()

def main():
    #SANITY CHECK
    m = RandomMDP(7, 2)
    m.transitions = np.array([[[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]],

                              [[0.9, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
                               [0.1, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0]],

                              [[0.0, 0.9, 0., 0.1, 0.0, 0.0, 0.0],
                               [0.0, 0.1, 0.0, 0.9, 0.0, 0.0, 0.0]],

                              [[0.0, 0.0, 0.9, 0.0, 0.1, 0.0, 0.0],
                               [0.0, 0.0, 0.1, 0.0, 0.9, 0.0, 0.0]],

                              [[0.0, 0.0, 0.0, 0.9, 0.0, 0.1, 0.0],
                               [0.0, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0]],

                              [[0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.1],
                               [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.9]],

                              [[0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9]]
                              ])
    m.rewards = [0., 1., 1., -5., 1., 1., 0.]
    print m.solve()
    print m
    print m.stationary_dist("1111111")


if __name__ == '__main__':
    main()

