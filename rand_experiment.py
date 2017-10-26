import random_MDP
import sys

# Default RMDP Params #
#######################
NUM_STATES = 7
NUM_ACTIONS = 2
SEED = None

def main(argv):
    num_states = argv[1] if len(argv) > 1 else NUM_STATES
    num_actions = argv[2] if len(argv) > 2 else NUM_ACTIONS
    seed = argv[3] if len(argv) > 3 else SEED

    rmdp = random_MDP.RandomMDP(num_states, num_actions, seed)

    pass

if __name__ == '__main__':
    main(sys.argv[1:])
