import tensorflow as tf
import numpy as np
from scipy.optimize import linprog
import argparse
import sys

NUMBER_OF_STATES = 7

GRAD_DESCENT_ITERATIONS = 10 ** 5

GREEDY_POLICY_ITERATIONS = 1

# Probability of choosing the move favored by the policy
POLICY_OBEY_RATE = 0.9

DISCOUNT_FACTOR = 0.75

OPPOSITE_ACTION_RATE = 0.1
DISCOUNT_FACTOR = 0.9

REWARDS = [0., 1., 1., -5., 1., 1., 0.]

# Setup for Q-Learning Neural Net

# Numbers used to refer to each state
STATES = tf.constant([0., 1., 2., 3., 4., 5., 6.])

state_action_distribution = tf.placeholder(tf.float32)

# Coefficients of polynomials for learned Q-values
initial_w = tf.truncated_normal([2, 4], stddev=0.1)
w = tf.Variable(initial_w)


# Computes the Q-value for a weight vector w and state
def q_value(w, state):
    return tf.reduce_sum(tf.multiply(w, [1, state, state ** 2, state ** 3]))


# Calculates Q-values from polynomial coefficients. The first row is for
#	moving right and the second is for left.
q_values = [
    [q_value(w[0], STATES[i]) for i in range(NUMBER_OF_STATES)],
    [q_value(w[1], STATES[i]) for i in range(NUMBER_OF_STATES)]]

# Maximum Q-value between 2 actions leaving each state
max_action_q_values = tf.reduce_max(q_values, 0)
# Estimated reward for an incoming state based on its reward value and best Q-value
next_state_reward = REWARDS + tf.scalar_mul(DISCOUNT_FACTOR, max_action_q_values)


def transition_prob(previous_state, right):
    prob = [0] * NUMBER_OF_STATES
    if right:
        prob[max(previous_state - 1, 0)] = OPPOSITE_ACTION_RATE
        prob[min(previous_state + 1, NUMBER_OF_STATES - 1)] = 1 - OPPOSITE_ACTION_RATE
    else:
        prob[max(previous_state - 1, 0)] = 1 - OPPOSITE_ACTION_RATE
        prob[min(previous_state + 1, NUMBER_OF_STATES - 1)] = OPPOSITE_ACTION_RATE
    return prob


# Transition probabilities for each state/action pair to new states
TRANSITION_PROBABILITIES = tf.constant([
    [transition_prob(i, False) for i in range(NUMBER_OF_STATES)],
    [transition_prob(i, True) for i in range(NUMBER_OF_STATES)]])

expected_future_reward = tf.reduce_sum(tf.multiply(TRANSITION_PROBABILITIES, next_state_reward), 2)

error = tf.reduce_sum(state_action_distribution * (q_values - expected_future_reward) ** 2, [0, 1])


# def state_action_dist_mixture(right_prop):
#     '''
#     Computes the state/action distribution in the convex hull of distributions
#     defined by the state/action distributions for moving left and right.
#
#     right_prop - number in [0,1] representing what proportion the right
#         distribution should have
#
#     returns - a list with an entry for the probability of being at each state
#     '''
#     return RIGHT_S_A_DIST * right_prop + LEFT_S_A_DIST * (1 - right_prop)


def get_optimal_weights(state_action_dist):
    '''
    Obtains the optimal weights for the q-values for some stationary distribution.

    state_action_dist - a matrix containing the distribution over states and actions

    returns - a 2x4 array of coefficients where rows correspond to moving left or right
        and columns correspond to the degree of polynomial
    '''
    # optimizer = tf.train.GradientDescentOptimizer(0.01)
    optimizer = tf.train.AdagradOptimizer(0.1)
    train = optimizer.minimize(error)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init, {state_action_distribution: state_action_dist})

    for i in range(GRAD_DESCENT_ITERATIONS):
        sess.run(train, {state_action_distribution: state_action_dist})

    # print("state action dist", sess.run(state_action_distribution, {state_action_distribution: state_action_dist}))
    # print("w", sess.run(w, {state_action_distribution: state_action_dist}))
    # print("Q-Values", sess.run(q_values, {state_action_distribution: state_action_dist}))
    # print("Max Q-Values", sess.run(max_action_q_values, {state_action_distribution: state_action_dist}))
    # print("Next State Reward", sess.run(next_state_reward, {state_action_distribution: state_action_dist}))
    # print("future reward by state/action", sess.run(tf.multiply(TRANSITION_PROBABILITIES, next_state_reward), {state_action_distribution: state_action_dist}))
    # print("expected_future_reward", sess.run(expected_future_reward, {state_action_distribution: state_action_dist}))
    # print("Error",sess.run(error, {state_action_distribution: state_action_dist}))
    return sess.run((w, q_values), {state_action_distribution: state_action_dist})


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
    return (linprog_out.x[0])

def get_next_state(action, current_state):
    '''
    Obtains the next state from a polynomial action function for some direction

    action - a lasso object which predicts the next state
    current_state - a real number representing the current state

    returns - a real number representing the new state
    '''
    return action.predict(np.array(current_state).reshape(-1, 1))


def weighted_error(state_action_dist, weight):
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
    return sess.run(error, {state_action_distribution: state_action_dist})


# def iteratively_learn_distributions(iterations):
#     '''
#     First, finds the Q-value weights for the stationary distributions defined
#     by always moving strictly left or right. Then, it computes the errors of
#     those weights. For each iteration, it finds the combination of
#     the left and right stationary distributions that yields the highest value
#     of the minimum of all errors so far. Then, it computes those errors and
#     adds it to the collection.
#
#     iterations - the number of weights to generate, excluding the
#         original hypotheses for left and right distributions.
#     '''
#     right_stationary_weight = get_optimal_weights(RIGHT_S_A_DIST)
#     left_stationary_weight = get_optimal_weights(LEFT_S_A_DIST)
#
#     right_weight_right_error = weighted_error(RIGHT_S_A_DIST, right_stationary_weight)
#     right_weight_left_error = weighted_error(LEFT_S_A_DIST, right_stationary_weight)
#     left_weight_right_error = weighted_error(RIGHT_S_A_DIST, left_stationary_weight)
#     left_weight_left_error = weighted_error(LEFT_S_A_DIST, left_stationary_weight)
#
#     left_errors = np.array([
#         [right_weight_left_error],
#         [left_weight_left_error]])
#     right_errors = np.array([
#         [right_weight_right_error],
#         [left_weight_right_error]])
#
#     for i in range(iterations):
#         new_weight = get_worst_weight(left_errors, right_errors)
#         new_state_action = state_action_dist_mixture(new_weight)
#
#         new_stationary_weight = get_optimal_weights(new_state_action)
#
#         new_weight_right_error = weighted_error(RIGHT_S_A_DIST, new_stationary_weight)
#         new_weight_left_error = weighted_error(LEFT_S_A_DIST, new_stationary_weight)
#
#         left_errors = np.append(
#             left_errors, new_weight_left_error)[np.newaxis].T
#         right_errors = np.append(
#             right_errors, new_weight_right_error)[np.newaxis].T


def get_stationary_vector(state, action_distribution):
    vec = [0] * NUMBER_OF_STATES
    if state == 0:
        vec[0] = action_distribution[0][0] - 1
        vec[1] = action_distribution[0][1]
    else:
        vec[state - 1] = action_distribution[1][state - 1]
        vec[state] = -1
        vec[state + 1] = action_distribution[0][state + 1]
    return vec


def find_greedy_policy(policy):
    '''
    For a given policy, optimizes the Q-values to minimize Least Squares Bellman
    error. Then, finds the policy minimizing error for those Q-values.
    '''
    #global w
    policy_array = np.asarray(policy)
    action_distribution = np.asarray([
        policy_array * (1 - POLICY_OBEY_RATE) + (1 - policy_array) * POLICY_OBEY_RATE,
        policy_array * POLICY_OBEY_RATE + (1 - policy_array) * (1 - POLICY_OBEY_RATE)])

    state_matrix = np.asarray(
        [get_stationary_vector(state, action_distribution) for state in range(NUMBER_OF_STATES - 1)]
        + [[1] * NUMBER_OF_STATES])
    # [0, 0, action_distribution[1][2], action_distribution[1][3] - 1]])
    state_distribution = np.transpose(
        np.matmul(np.linalg.inv(state_matrix), np.asarray([[0]] * (NUMBER_OF_STATES - 1) + [[1]])))

    state_action_dist = action_distribution * state_distribution  # dist induced by input p

    # print("State-action dist:", state_action_dist)

    train_policy = policy  # input
    train_s_a_dist = state_action_dist

    train_weights = get_optimal_weights(train_s_a_dist)  # w

    sess = tf.Session()
    sess.run(w.assign(train_weights[0]))
    q_vals = sess.run(q_values, {state_action_distribution: train_s_a_dist})  # est q vals
    err = sess.run(error, {state_action_distribution: train_s_a_dist})

    # print("Error:", err)

    greedy_policy = tuple(sess.run(tf.argmax(q_vals, 0)))  # resultant policy

    # print(train_policy, greedy_policy)

    q_out = []
    for q_list in q_vals:
        for each in q_list:
            q_out.append(str(each))
    w_out = []
    for w_list in train_weights[0]:
        for each in w_list:
            w_out.append(str(each))
    d_out = []
    for d_list in state_action_dist:
        for each in d_list:
            d_out.append(str(each))

    #format outputs for writing to csv
    outputs = [''.join([str(pol) for pol in train_policy]),
               ''.join([str(pol) for pol in greedy_policy]),
               str(err),
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

def main(args):
    policy = int(args[0])
    size = int(args[2])
    input_policy = [int(b) for b in format(policy, '0%db' % size)]
    print find_greedy_policy(input_policy)




if __name__ == '__main__':
    main(sys.argv[1:])
