import random_MDP as r
import general_q_estimation as qe
import sys


def main(args):
    num_states = int(args[1])
    num_actions = int(args[2])
    seed = int(args[3])
    input_pol = args[4]

    rmdp = r.RandomMDP(num_states,  num_actions, seed=seed)
    qe.find_greedy_policy(input_pol, rmdp.transitions, rmdp.rewards)
    return

def exp_file(num_states, num_actions, seed):
    with open(seed+"/exp"+seed+".txt", 'w') as f:
        f.write(r.RandomMDP(int(num_states),int(num_actions),int(seed)))
        f.close
    return

if __name__ == '__main__':
    main(sys.argv)