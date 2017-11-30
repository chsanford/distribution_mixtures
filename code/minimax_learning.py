import tensorflow as tf
import numpy as np
from scipy.optimize import linprog
import argparse
import sys
import matplotlib.pyplot as plt


GRAD_DESCENT_ITERATIONS = 10 ** 5 # this may need tuning
POLYNOMIAL_DEGREE = 3
POLICY_OBEY_RATE = 0.9
OPPOSITE_ACTION_RATE = 0.1
DISCOUNT_FACTOR = 0.9
NUMBER_STATES = 4
REWARD = np.asarray([0,1,1,0])
TRANSITION = np.asarray([
    [[0.9, 0.1, 0, 0], [0.1, 0.9, 0, 0]],
    [[0.9, 0, 0.1, 0], [0.1, 0, 0.9, 0]],
    [[0, 0.9, 0, 0.1], [0, 0.1, 0, 0.9]],
    [[0, 0, 0.9, 0.1], [0, 0, 0.1, 0.9]]])

## Set up q-values computation from weights and states
state_action_dist = tf.placeholder(tf.float32, shape=(2, NUMBER_STATES))


# Numbers used to refer to each state
states = range(NUMBER_STATES)

# Coefficients of polynomials for learned Q-values
initial_w = tf.truncated_normal([2, POLYNOMIAL_DEGREE + 1], stddev=0.1)
w = tf.Variable(initial_w)

# Each state raised to every polynomial degree
state_basis_matrix = tf.constant(
    [[state ** power for state in states] for power in range(POLYNOMIAL_DEGREE + 1)],
    dtype=tf.float32)

# Calculates Q-values from polynomial coefficients. The first row is for
#   moving right and the second is for left.
q_values = tf.matmul(w, state_basis_matrix)

# Maximum Q-value between 2 actions leaving each state
max_action_q_values = tf.reduce_max(q_values, 0)

## Set up neural net used to find error

# Estimated reward for an incoming state based on its reward value and best Q-value
next_state_reward = REWARD + tf.scalar_mul(DISCOUNT_FACTOR, max_action_q_values)

# For each state and action, expected future reward of that choice
transition_tensor = tf.constant(TRANSITION, dtype=tf.float32)
expected_future_reward = tf.transpose(
    tf.reduce_sum(tf.multiply(transition_tensor, next_state_reward), 2))

# Squared Bellman error for policy
error = tf.reduce_sum(state_action_dist *
    (q_values - expected_future_reward) ** 2, [0, 1])


def find_q_values(state_action_distribution):
    ## Set up q-values computation from weights and states

    ## Obtains optimal weights with neural net structure

    optimizer = tf.train.AdagradOptimizer(0.1)
    train = optimizer.minimize(error)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(GRAD_DESCENT_ITERATIONS):
        sess.run(train, {state_action_dist: state_action_distribution})

    # print("state action dist", state_action_dist)
    # print("w", sess.run(w))
    # print("Q-Values", sess.run(q_values))
    # print("Max Q-Values", sess.run(max_action_q_values))
    # print("Next State Reward", sess.run(next_state_reward))
    # print("expected_future_reward", sess.run(expected_future_reward))
    # print("Error",sess.run(error))

    w_val = sess.run(w, {state_action_dist: state_action_distribution})
    q_value_val = sess.run(q_values, {state_action_dist: state_action_distribution})
    error_val = sess.run(error, {state_action_dist: state_action_distribution})

    return(q_value_val, w_val, error_val)

def get_state_action_dist(policy):

    # Probability of choosing each action
    action_dist = np.asarray([
        policy * (1 - POLICY_OBEY_RATE) + (1 - policy) * POLICY_OBEY_RATE,
        policy * POLICY_OBEY_RATE + (1 - policy) * (1 - POLICY_OBEY_RATE)])

    # Probability of choosing each state from a prior state given the policy
    next_state_dist = []
    for state_index in range(NUMBER_STATES):
        next_state_dist.append(
            np.multiply(action_dist[0][state_index], TRANSITION[state_index][0]) +
                np.multiply(action_dist[1][state_index], TRANSITION[state_index][1]))
    next_state_dist = np.transpose(np.asarray(next_state_dist))

    # Matrix used to find probabilities of the stationary distribution
    stationary_matrix = next_state_dist - np.identity(NUMBER_STATES)
    stationary_matrix[NUMBER_STATES - 1] = np.asarray([1] * NUMBER_STATES)

    # Stationary distribution over states given policy
    state_dist = np.transpose(
        np.matmul(np.linalg.inv(stationary_matrix), np.asarray([[0]] * (NUMBER_STATES - 1) + [[1]])))

    # Distribution of state-action pairs
    state_action_dist = action_dist * state_dist

    return state_action_dist

def find_greedy_policy(initial_policy):
    '''
    For a given policy, optimizes the Q-values to minimize Least Squares Bellman
    error. Then, finds the policy minimizing error for those Q-values.

    initial_policy - an array of 0/1 indicating the initial policy to take for each state
    transition - an array indexed by (s,a,s') such that each entry represents the
        probability of tranditioning from s to s' with action a
    reward - an array representing reward for each state
    '''

    init_pol = np.asarray([float(p) for p in initial_policy])

    state_action_dist = get_state_action_dist(init_pol)

    (q_value_val, w_val, error_val) = find_q_values(state_action_dist)

    greedy_policy = tuple(np.argmax(q_value_val, 0))  # resultant policy

    q_out = []
    for q_list in q_value_val:
        for each in q_list:
            q_out.append(str(each))
    w_out = []
    for w_list in w_val:
        for each in w_list:
            w_out.append(str(each))
    d_out = []
    for d_list in state_action_dist:
        for each in d_list:
            d_out.append(str(each))

    #format outputs for writing to csv
    outputs = [''.join([str(pol) for pol in initial_policy]),
               ''.join([str(pol) for pol in greedy_policy]),
               str(error_val),
               ', '.join(q_out),
               ', '.join(w_out),
               ', '.join(d_out)]

    # print "Train policy: ", outputs[0]
    # print "Greedy policy: ", outputs[1]
    # print "Error: ", outputs[2]
    # print "Qs: ", outputs[3]
    # print "Ws: ", outputs[4]
    # print "Ds: ", outputs[5]
    return ', '.join(outputs)

def weighted_error(state_action_distribution, weight):
    '''
    Computes the weighted Least Squares Bellman Error based on a stationary
    distribution.

    state_action_dist - distribution of states and actions
    weight - 2x3 matrix representing coefficients

    returns - a real number >= represting the error weighted by the stationary
        distribution
    '''
    sess = tf.Session()
    sess.run(w.assign(weight))
    return sess.run(error, {state_action_dist: state_action_distribution})

def weighted_q_val_error(q_val, opt_q_val, weight):
    '''
    Computes the squared difference of some Q-values and the optimal Q-values,
    where terms combined with weights based on the state-action distribution.
    '''
    return sum(sum(weight * (q_val - opt_q_val) ** 2))

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
    for i in range(len(left_errors)):
        plt.plot([0,1], [left_errors[i], right_errors[i]])
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
    return(linprog_out.x[0])

def state_action_dist_mixture(sa_dist1, sa_dist2, proportion1):
    '''
    Computes the state/action distribution in the convex hull of distributions
    defined by the state/action distributions for using each action.

    proportion1 - number in [0,1] representing what proportion the first
        distribution should have

    returns - a list with an entry for the probability of being at each state
    '''
    return sa_dist1 * proportion1 + sa_dist2 * (1 - proportion1)

def iteratively_learn_distributions(iterations, stationary_dist1, stationary_dist2):
    '''
    First, finds the Q-value weights for the two input stationary distributions. 
    Then, it computes the errors of
    those weights. For each iteration, it finds the combination of
    those stationary distributions that yields the highest value
    of the minimum of all errors so far. Then, it computes those errors and
    adds it to the collection.

    iterations - the number of weights to generate, excluding the
        original hypotheses for the distributions.
    stationary_dist1 - a matrix representing the state-action dist of action 1
    stationary_dist2 - a matrix representing the state-action dist of action 2
    '''
    q_vals1, stationary_weight1, _ = find_q_values(stationary_dist1)
    q_vals2, stationary_weight2, _ = find_q_values(stationary_dist2)

    print(q_vals1)
    print(stationary_dist1)

    weight1_error1 = weighted_error(stationary_dist1, stationary_weight1)
    weight1_error2 = weighted_error(stationary_dist2, stationary_weight1)
    weight2_error1 = weighted_error(stationary_dist1, stationary_weight2)
    weight2_error2 = weighted_error(stationary_dist2, stationary_weight2)

    ## Uncomment below code for new errors
    # opt_q_vals = # (# actions) x (# states) numpy array
    # weight1_error1 = weighted_q_val_error(q_vals1, opt_q_vals, stationary_dist1)
    # weight1_error2 = weighted_q_val_error(q_vals2, opt_q_vals, stationary_dist1)
    # weight2_error1 = weighted_q_val_error(q_vals1, opt_q_vals, stationary_dist2)
    # weight2_error2 = weighted_q_val_error(q_vals2, opt_q_vals, stationary_dist2)

    errors2 = np.array([
        [weight1_error2],
        [weight2_error2]])
    errors1 = np.array([
        [weight1_error1],
        [weight2_error1]])

    visualize_hypothesis_errors(errors1, errors2)

    for i in range(iterations):
        print "errors2", errors2
        print "errors1", errors1
        new_weight = get_worst_weight(errors2, errors1)
        new_stationary_dist = state_action_dist_mixture(stationary_dist1, stationary_dist2, new_weight)

        print(new_weight)
        print(new_stationary_dist)

        new_q_vals, new_stationary_weight, _ = find_q_values(new_stationary_dist)

        new_weight_error1 = weighted_error(stationary_dist1, new_stationary_weight)
        new_weight_error2 = weighted_error(stationary_dist2, new_stationary_weight)

        ## Uncomment for new errors
        # new_weight_error1 = weighted_q_val_error(new_q_vals, opt_q_vals, stationary_dist1)
        # new_weight_error2 = weighted_q_val_error(new_q_vals, opt_q_vals, stationary_dist2)

        errors2 = np.append(
            errors2, new_weight_error2)[np.newaxis].T
        errors1 = np.append(
            errors1, new_weight_error1)[np.newaxis].T

        visualize_hypothesis_errors(errors1, errors2)

#find_greedy_policy(np.asarray([0, 0, 0, 0]), transition, np.asarray([0, 1, 1, 0]))

left_stationary = get_state_action_dist(np.asarray([0,0,0,0]))
right_stationary = get_state_action_dist(np.asarray([1,1,1,1]))
iteratively_learn_distributions(5, left_stationary, right_stationary)


