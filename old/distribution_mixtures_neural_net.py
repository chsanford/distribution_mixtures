import tensorflow as tf
import random
import math
import collections
import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import LassoLarsIC
import matplotlib.pyplot as plt

NUMBER_OF_STATES = 4

GRAD_DESCENT_ITERATIONS = 10 ** 5

GREEDY_POLICY_ITERATIONS = 100

# Probability of choosing the move not favored by the policy

#### RENAME THIS
EXPLORATION_RATE = 0.9

# Probability of moving the opposite direction that the action indicates
OPPOSITE_ACTION_RATE = 0.1
STATIONARY_DIST_PROPORTION = [
	OPPOSITE_ACTION_RATE ** 3,
	OPPOSITE_ACTION_RATE ** 2 * (1 - OPPOSITE_ACTION_RATE),
	OPPOSITE_ACTION_RATE * (1 - OPPOSITE_ACTION_RATE) ** 2,
	(1 - OPPOSITE_ACTION_RATE) ** 3]
RIGHT_STATIONARY_ARRAY = [
	p / sum(STATIONARY_DIST_PROPORTION) for p in STATIONARY_DIST_PROPORTION]

RIGHT_STATIONARY_DIST = np.asarray(RIGHT_STATIONARY_ARRAY)
LEFT_STATIONARY_DIST = RIGHT_STATIONARY_DIST[::-1]

RIGHT_S_A_DIST = np.asarray([[0.1], [0.9]]) * RIGHT_STATIONARY_DIST
LEFT_S_A_DIST = np.asarray([[0.9], [0.1]]) * LEFT_STATIONARY_DIST

TRUE_RIGHT_ACTION = np.asarray([1, 2, 3, 3])
TRUE_LEFT_ACTION = np.asarray([0, 0, 1, 2])

REWARD_POLYNOMIAL = [0, 1.5, -0.5]

DISCOUNT_FACTOR = 0.75

MIN_REWARD = 0
MAX_REWARD = 1.0 / (1 - DISCOUNT_FACTOR)

OPPOSITE_ACTION_RATE = 0.1
DISCOUNT_FACTOR = 0.9

## Setup for Q-Learning Neural Net

# Numbers used to refer to each state
STATES = tf.constant([0., 1., 2., 3.])
# Reward for arriving at each state
REWARDS = tf.constant([0., 1., 1., 0.])

state_action_distribution = tf.placeholder(tf.float32)

# Actions taken for each state; left if 0, right if 1
policy = tf.constant([1., 1., 0., 0.])
# Generates a matrix where each column represents the probability of choosing
#	the two actions for each state based on the policy
exploration_policy = [
	tf.multiply(policy, 1 - EXPLORATION_RATE) + tf.multiply(1 - policy, EXPLORATION_RATE),
	tf.multiply(1 - policy, 1 - EXPLORATION_RATE) + tf.multiply(policy, EXPLORATION_RATE)]
# Combined probability of choosing each state and action
# state_action_distribution = stationary_distribution * exploration_policy

# Coefficients of polynomials for learned Q-values
initial_w = tf.truncated_normal([2, 3], stddev=0.1)
w = tf.Variable(initial_w)

# Computes the Q-value for a weight vector w and state
def q_value(w, state):
	return tf.reduce_sum(tf.multiply(w, [1, state, state ** 2]))

# Calculates Q-values from polynomial coefficients. The first row is for
#	moving right and the second is for left.
q_values = [
	[q_value(w[0], STATES[0]), q_value(w[0], STATES[1]), q_value(w[0], STATES[2]), q_value(w[0], STATES[3])],
	[q_value(w[1], STATES[0]), q_value(w[1], STATES[1]), q_value(w[1], STATES[2]), q_value(w[1], STATES[3])]]

# Maximum Q-value between 2 actions leaving each state
max_action_q_values = tf.reduce_max(q_values, 0)
# Estimated reward for an incoming state based on its reward value and best Q-value
next_state_reward = REWARDS + tf.scalar_mul(DISCOUNT_FACTOR, max_action_q_values)

# Transition probabilities for each state/action pair to new states
TRANSITION_PROBABILITIES = tf.constant([
	[
		[1. - OPPOSITE_ACTION_RATE, OPPOSITE_ACTION_RATE, 0., 0.],
		[1. - OPPOSITE_ACTION_RATE, 0., OPPOSITE_ACTION_RATE, 0.],
		[0., 1. - OPPOSITE_ACTION_RATE, 0., OPPOSITE_ACTION_RATE],
		[0., 0., 1. - OPPOSITE_ACTION_RATE, OPPOSITE_ACTION_RATE]],
	[
		[OPPOSITE_ACTION_RATE, 1. - OPPOSITE_ACTION_RATE, 0., 0.],
		[OPPOSITE_ACTION_RATE, 0., 1. - OPPOSITE_ACTION_RATE, 0.],
		[0., OPPOSITE_ACTION_RATE, 0., 1. - OPPOSITE_ACTION_RATE],
		[0., 0., OPPOSITE_ACTION_RATE, 1. - OPPOSITE_ACTION_RATE]]])

expected_future_reward = tf.reduce_sum(tf.multiply(TRANSITION_PROBABILITIES, next_state_reward), 2)

error = tf.reduce_sum(state_action_distribution * (q_values - expected_future_reward) ** 2, [0, 1])


def state_action_dist_mixture(right_prop):
	'''
	Computes the state/action distribution in the convex hull of distributions
	defined by the state/action distributions for moving left and right.

	right_prop - number in [0,1] representing what proportion the right
		distribution should have

	returns - a list with an entry for the probability of being at each state
	'''
	return RIGHT_S_A_DIST * right_prop + LEFT_S_A_DIST * (1 - right_prop)

def get_optimal_weights(state_action_dist):
	'''
	Obtains the optimal weights for the q-values for some stationary distribution.

	state_action_dist - a matrix containing the distribution over states and actions

	returns - a 2x3 array of coefficients where rows correspond to moving left or right 
		and columns correspond to the degree of polynomial
	'''
	#optimizer = tf.train.GradientDescentOptimizer(0.01)
	optimizer = tf.train.AdagradOptimizer(0.01)
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
	return sess.run(w, {state_action_distribution: state_action_dist})
	
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

def iteratively_learn_distributions(iterations):
	'''
	First, finds the Q-value weights for the stationary distributions defined
	by always moving strictly left or right. Then, it computes the errors of
	those weights. For each iteration, it finds the combination of
	the left and right stationary distributions that yields the highest value
	of the minimum of all errors so far. Then, it computes those errors and
	adds it to the collection.

	iterations - the number of weights to generate, excluding the
		original hypotheses for left and right distributions.
	''' 
	right_stationary_weight = get_optimal_weights(RIGHT_S_A_DIST)
	left_stationary_weight = get_optimal_weights(LEFT_S_A_DIST)

	right_weight_right_error = weighted_error(RIGHT_S_A_DIST, right_stationary_weight)
	right_weight_left_error = weighted_error(LEFT_S_A_DIST, right_stationary_weight)
	left_weight_right_error = weighted_error(RIGHT_S_A_DIST, left_stationary_weight)
	left_weight_left_error = weighted_error(LEFT_S_A_DIST, left_stationary_weight)

	left_errors = np.array([
		[right_weight_left_error],
		[left_weight_left_error]])
	right_errors = np.array([
		[right_weight_right_error],
		[left_weight_right_error]])

	visualize_hypothesis_errors(left_errors, right_errors)

	for i in range(iterations):
		new_weight = get_worst_weight(left_errors, right_errors)
		new_state_action = state_action_dist_mixture(new_weight)

		new_stationary_weight = get_optimal_weights(new_state_action)

		new_weight_right_error = weighted_error(RIGHT_S_A_DIST, new_stationary_weight)
		new_weight_left_error = weighted_error(LEFT_S_A_DIST, new_stationary_weight)

		left_errors = np.append(
			left_errors, new_weight_left_error)[np.newaxis].T
		right_errors = np.append(
			right_errors, new_weight_right_error)[np.newaxis].T

		visualize_hypothesis_errors(left_errors, right_errors)

def find_greedy_policies():
	'''
	For each policy, optimizes the Q-values to minimize Least Squares Bellman
	error. Then, finds the policy minimizing error for those Q-values.

	returns - a dictionary mapping each policy to the policy optimal for its
		Q-values.
	'''
	policies = [(
		i % 2,
		math.floor(i / 2) % 2,
		math.floor(i / 4) % 2,
		math.floor(i / 8) % 2) 
		for i in range(16)]

	state_action_distributions = []
	greedy_policies = {policy: collections.Counter() for policy in policies}

	for policy in policies:
		policy_array = np.asarray(policy)
		action_distribution = np.asarray([
			policy_array * (1 - EXPLORATION_RATE) + (1 - policy_array) * EXPLORATION_RATE,
			policy_array * EXPLORATION_RATE + (1 - policy_array) * (1 - EXPLORATION_RATE)])

		state_matrix = np.asarray([
			[action_distribution[0][0] - 1, action_distribution[0][1], 0, 0],
			[action_distribution[1][0], -1, action_distribution[0][2], 0],
			[0, action_distribution[1][1], -1, action_distribution[0][3]],
			[1, 1, 1, 1]])
			#[0, 0, action_distribution[1][2], action_distribution[1][3] - 1]])
		state_distribution = np.transpose(
			np.matmul(np.linalg.inv(state_matrix), np.asarray([[0], [0], [0], [1]])))

		state_action_dist = action_distribution * state_distribution
		state_action_distributions.append(state_action_dist)

	for i in range(len(policies)):
		train_policy = policies[i]
		train_s_a_dist = state_action_distributions[i]

		for i in range(GREEDY_POLICY_ITERATIONS):
			train_weights = get_optimal_weights(train_s_a_dist)

			sess = tf.Session()
			sess.run(w.assign(train_weights))
			q_vals = sess.run(q_values, {state_action_distribution: train_s_a_dist})
			err = sess.run(error, {state_action_distribution: train_s_a_dist})

			print(tuple(sess.run(tf.argmax(q_vals, 0))), err)

			if i == 0 or err < opt_err:
				opt_err = err
				opt_q_vals = q_vals

		greedy_policy = tuple(sess.run(tf.argmax(opt_q_vals, 0)))
		#greedy_policies[train_policy][greedy_policy] += 1
		print(train_policy, greedy_policy)


#iteratively_learn_distributions(3)
find_greedy_policies()


