import tensorflow as tf
import numpy as np
from scipy.optimize import linprog
import argparse
import sys

GRAD_DESCENT_ITERATIONS = 10 ** 5
POLYNOMIAL_DEGREE = 3
POLICY_OBEY_RATE = 0.9
OPPOSITE_ACTION_RATE = 0.1
DISCOUNT_FACTOR = 0.9

def find_greedy_policy(initial_policy, transition, reward):
    '''
    For a given policy, optimizes the Q-values to minimize Least Squares Bellman
    error. Then, finds the policy minimizing error for those Q-values.

    initial_policy - an array of 0/1 indicating the initial policy to take for each state
    transition - an array indexed by (s,a,s') such that each entry represents the
        probability of tranditioning from s to s' with action a
    reward - an array representing reward for each state
    '''

    ## Set up q-values computation from weights and states

    number_states = len(initial_policy)
    # Numbers used to refer to each state
    states = range(number_states)

    # Coefficients of polynomials for learned Q-values
    initial_w = tf.truncated_normal([2, POLYNOMIAL_DEGREE + 1], stddev=0.1)
    w = tf.Variable(initial_w)

    # Each state raised to every polynomial degree
    state_basis_matrix = tf.constant(
        [[state ** power for state in states] for power in range(POLYNOMIAL_DEGREE + 1)],
        dtype=tf.float32)

    # Calculates Q-values from polynomial coefficients. The first row is for
    #   moving right and the second is for left.
    q_values = tf.matmul(w, tf.transpose(state_basis_matrix))

    # Maximum Q-value between 2 actions leaving each state
    max_action_q_values = tf.reduce_max(q_values, 0)


    ## Set up transition probabilities for the policy

    # Probability of choosing each action
    action_dist = np.asarray([
        initial_policy * (1 - POLICY_OBEY_RATE) + (1 - initial_policy) * POLICY_OBEY_RATE,
        initial_policy * POLICY_OBEY_RATE + (1 - initial_policy) * (1 - POLICY_OBEY_RATE)])

    # Probability of choosing each state from a prior state given the policy
    next_state_dist = []
    for state_index in range(number_states):
        next_state_dist.append(
            np.multiply(action_dist[0][state_index], transition[state_index][0]) +
                np.multiply(action_dist[1][state_index], transition[state_index][1]))
    next_state_dist = np.transpose(np.asarray(next_state_dist))


    # Matrix used to find probabilities of the stationary distribution
    stationary_matrix = next_state_dist - np.identity(number_states)
    stationary_matrix[number_states - 1] = np.asarray([1] * number_states)

    # Stationary distribution over states given policy
    state_dist = np.transpose(
        np.matmul(np.linalg.inv(stationary_matrix), np.asarray([[0]] * (number_states - 1) + [[1]])))

    # Distribution of state-action pairs
    state_action_dist = action_dist * state_dist


    ## Set up neural net used to find error

    # Estimated reward for an incoming state based on its reward value and best Q-value
    next_state_reward = reward + tf.scalar_mul(DISCOUNT_FACTOR, max_action_q_values)

    # For each state and action, expected future reward of that choice
    transition_tensor = tf.constant(transition, dtype=tf.float32)
    expected_future_reward = tf.transpose(
        tf.reduce_sum(tf.multiply(transition_tensor, next_state_reward), 2))

    # Squared Bellman error for policy
    error = tf.reduce_sum(state_action_dist *
        (q_values - expected_future_reward) ** 2, [0, 1])


    ## Obtains optimal weights with neural net structure

    optimizer = tf.train.AdagradOptimizer(0.1)
    train = optimizer.minimize(error)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(GRAD_DESCENT_ITERATIONS):
        sess.run(train)

    print("state action dist", state_action_dist)
    print("w", sess.run(w))
    print("Q-Values", sess.run(q_values))
    print("Max Q-Values", sess.run(max_action_q_values))
    print("Next State Reward", sess.run(next_state_reward))
    print("expected_future_reward", sess.run(expected_future_reward))
    print("Error",sess.run(error))

    w_val = sess.run(w)
    q_value_val = sess.run(q_values)
    error_val = sess.run(error)

    greedy_policy = tuple(sess.run(tf.argmax(q_values, 0)))  # resultant policy

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

transition = np.asarray([
    [[0.9, 0.1, 0, 0], [0.1, 0.9, 0, 0]],
    [[0.9, 0, 0.1, 0], [0.1, 0, 0.9, 0]],
    [[0, 0.9, 0, 0.1], [0, 0.1, 0, 0.9]],
    [[0, 0, 0.9, 0.1], [0, 0, 0.1, 0.9]]])

find_greedy_policy(np.asarray([0, 0, 0, 0]), transition, np.asarray([0, 1, 1, 0]))
