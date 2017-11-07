import random_MDP as r
import sys


def main(args):
    import general_q_estimation as qe

    num_states = int(args[1])
    num_actions = int(args[2])
    deg = int(args[3])
    seed = int(args[4])
    input_pol = [int(b) for b in format(int(args[5]), '0%db' % num_states)] if len(args) > 5 else []

    rmdp = r.RandomMDP(num_states,  num_actions, seed=seed)
    if not input_pol:
        input_pol = rmdp.p_opt
    for pol in input_pol:
        res = qe.find_greedy_policy(pol, rmdp.transitions, rmdp.rewards, poly_deg=deg)
        if res.split(',')[1].strip() in [p.strip() for p in rmdp.p_opt]:
            print res + ", optimal\n"
        else:
            print res + ", suboptimal\n"
    return


def exp_file(num_states, num_actions, seed):
    with open(str(seed)+"/exp"+str(seed)+".txt", 'w') as f:
        f.write(str(r.RandomMDP(int(num_states),int(num_actions),seed=int(seed))))
        f.close
    return


if __name__ == '__main__':
    main(sys.argv)