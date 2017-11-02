import random_MDP as r
import general_q_estimation as qe
import sys


def main(args):
    num_states = int(args[1])
    num_actions = int(args[2])
    seed = int(args[3])
    input_pol = [int(b) for b in format(int(args[4]), '0%db' % num_states)] if len(args)>4 else []


    rmdp = r.RandomMDP(num_states,  num_actions, seed=seed)
    if not input_pol:
        input_pol = rmdp.p_opt[0]
    print qe.find_greedy_policy(input_pol, rmdp.transitions, rmdp.rewards)
    return

def exp_file(num_states, num_actions, seed):
    with open(seed+"/exp"+seed+".txt", 'w') as f:
        f.write(r.RandomMDP(int(num_states),int(num_actions),int(seed)))
        f.close
    return

if __name__ == '__main__':
    main(sys.argv)