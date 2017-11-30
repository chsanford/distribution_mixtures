import numpy as np
import random_MDP as rmdp
import matplotlib.pyplot as plt
from scipy.optimize import linprog

sample_size = 10000
num_states = 7
num_actions = 2
dim = 3
num_features = dim * num_actions  # k

gamma = 0.9
mdp = rmdp.RandomMDP(num_states, num_actions, seed=12)
opt_pol = mdp.p_opt[0]


def main():
    left_pol = ''.join([str(np.random.choice(num_actions)) for _ in range(num_states)])
    right_pol = ''.join([str(np.random.choice(num_actions)) for _ in range(num_states)])

    print "Using policies: ",
    print left_pol,
    print right_pol
    left_dist = mdp.state_action_dist(left_pol)
    right_dist = mdp.state_action_dist(right_pol)

    print "Left dist:"
    print left_dist
    print "Right dist:"
    print right_dist

    left_sample = mdp.sample(sample_size, dist=left_dist)
    right_sample = mdp.sample(sample_size, dist=right_dist)

    print "Opt pol: ", opt_pol
    print "get_error(left_sample, left_dist, opt_pol)"
    left_errs = np.asarray(get_error(left_sample, left_dist, opt_pol))
    print "get_error(left_sample, right_dist, opt_pol)"
    right_errs = np.asarray(get_error(left_sample, right_dist, opt_pol))

    print "get_error(right_sample, left_dist, opt_pol)"
    left_errs = np.append(left_errs, get_error(right_sample, left_dist, opt_pol), axis=0)
    print "get_error(right_sample, right_dist, opt_pol)"
    right_errs = np.append(right_errs, get_error(right_sample, right_dist, opt_pol), axis=0)
    wt = get_worst_weight(left_errs, right_errs)
    print wt
    # wt = 0.87
    print "mix_wt: ", wt
    print left_errs
    print right_errs
    visualize_hypothesis_errors(left_errs, right_errs)

    mix_dist = (1 - wt) * left_dist + wt * right_dist
    print "On mix: "
    left_errs = np.append(left_errs, get_error(left_sample, mix_dist, opt_pol), axis=0)
    right_errs = np.append(right_errs, get_error(right_sample, mix_dist, opt_pol), axis=0)
    print "mix_wt: ", wt

    visualize_hypothesis_errors(left_errs, right_errs)


def visualize_hypothesis_errors(left_errors, right_errors):
    '''
    Plots a visualization of the errors of hypotheses for combinations of
    the right and left stationary distributions.

    left_errors - a numpy column vector where the ith entry represents the
        error of the ith hypothesis with respect to the left stationary
        distribution.
    right_errors - a numpy column vector like above, except wrt the right
        stationary distribution.

    returns - a real number in [0,1] representing the proportion of the
        right stationary distribution in the distribution with the highest
        error.
    '''
    # for i in range(len(left_errors)):
    handles = [handle for handle in [plt.plot([0, 1], [left_errors[i], right_errors[i]]) for i in
                                     range(len(left_errors))]]
    plt.legend(labels=range(len(left_errors)))
    plt.show()


def get_worst_weight(left_errors, right_errors):
    '''
    Uses linear programming to determine the ratio of the combination of the
    right and left stationary distributions with the maximum value for the
    minimum errors between the estimators so far.

    left_errors - a numpy column vector where the ith entry represents the
        error of the ith hypothesis with respect to the left stationary
        distribution.
    right_errors - a numpy column vector like above, except wrt the right
        stationary distribution.

    returns - a real number in [0,1] representing the proportion of the
        right stationary distribution in the distribution with the highest
        error.
    '''
    c = np.array([0, -1])
    b_ub = np.concatenate((left_errors, np.array([[1], [0]])))
    A_ub = np.concatenate((left_errors - right_errors,
                           np.ones(len(left_errors))[np.newaxis].T), 1)
    A_ub = np.concatenate((A_ub, np.array([[1, 0], [-1, 0]])))
    linprog_out = linprog(c=c, A_ub=A_ub, b_ub=b_ub)
    return linprog_out.x[0]


def phi(state, action):
    # return np.asarray([1.0 if i == state or i == num_states + action else 0.0 for i in range(num_features)])
    features = [0] * num_features
    features[action * dim:(action + 1) * dim] = [state ** i for i in range(dim)]
    return np.transpose(np.asarray([features]))


def get_weighted_residual(q_est, target_pi, dist):
    print dist.shape
    q_hat = np.asarray([[q_est(s, a)] for s in range(num_states) for a in range(num_actions)])
    return


def estimate_q():
    pass

def get_error(sample, dist, error_pol):
    # Given a sample, estimate Q*
    A = np.zeros((num_features, num_features))
    b = np.zeros((num_features, 1))
    for (s, a, r, sp) in sample:
        A += np.dot(phi(s, a), np.transpose(phi(s, a) - gamma * phi(sp, int(error_pol[sp]))))
        b += phi(s, a) * r
    w = np.dot(np.linalg.inv(A), b)
    q_est = (lambda state, action: np.dot(np.transpose(phi(state, action)), w))

    # Compute MSE of q_est to Q*
    err = 0.0
    for state in range(num_states):
        for action in range(num_actions):
            err += dist[state][action] * (q_est(state, action) - mdp.q_star[state, action]) ** 2
            # err += np.square(q_est(state, action) - mdp.q_star[state, action])
    err = err / (num_states * num_actions)
    print "MSE: ", err
    return err


if __name__ == '__main__':
    main()