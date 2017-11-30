import pandas as pd
import sys

def find_cycles(policy_dict):
    policies = policy_dict.keys()
    grouped_policies = []
    cycles = []
    while policies:
        start_policy = policies.pop()
        chain = [start_policy]
        cyclic = False
        while not cyclic:
            next_policy = policy_dict[chain[-1]]
            for group in grouped_policies:
                if next_policy in group:
                    group.update(set(chain))
                    break;
            if next_policy in chain:
                cyclic = True
                cycles.append(chain[chain.index(next_policy):])
                grouped_policies.append(set(chain))
            elif next_policy not in policies:
                break;
            else:
                chain.append(next_policy)
                policies.remove(next_policy)
    return (grouped_policies, cycles)

def is_opt_stable(cycles, opt):
    return [opt] in cycles

def exist_cycles_without_opt(cycles, opt):
    return any([(opt not in cycle and len(cycle) > 1) for cycle in cycles])

def main(args):
    # Reads the MDP data from a CSV with the given headers and loads it into a dataframe
    path = args[0]
    input_path = path + "results.csv"
    names = ["train_policy", "greedy_policy", "err", "q_L0", "q_L1", "q_L2", "q_L3",
        "q_R0", "q_R1", "q_R2", "q_R3", "w_L0", "w_L1", "w_L2", "w_R0", "w_R1", "w_R2",
        "d_L0", "d_L1", "d_L2", "d_L3", "d_R0", "d_R1", "d_R2", "d_R3", "opt"]
    mdp_data = pd.read_csv(input_path, names=names, skiprows=[0], index_col=None)

    # Filters data to be only results for each train policy with min error 
    min_err_id = mdp_data.groupby(["train_policy"])['err'].idxmin()
    min_err_mdp_data = mdp_data.loc[min_err_id]

    # Gets maps of initial policy to greedy policy and of whether each policy is optimal
    policy_map = dict(zip(min_err_mdp_data.train_policy, min_err_mdp_data.greedy_policy))

    # Creates output file
    f = open(path + "summary.txt", "w+")

    # Finds cycles and disjoint grouped policies and outputs data
    (grouped_policies, cycles) = find_cycles(policy_map)
    for i in range(len(grouped_policies)):
        f.write("Group " + str(i) + ": " + str(list(grouped_policies[i])) + "\n")
        f.write("Stable Cycle " + str(i) + ": " + str(cycles[i]) + "\n")

    # Determines which optimal policies are stable, cyclic, or unstable
    stable_opt = []
    cyclic_opt = []
    unstable_opt = []
    
    for i, row in min_err_mdp_data.iterrows():
      train_policy = row["train_policy"]
      if row["opt"] == " optimal":
            unstable_opt.append(train_policy)
            for cycle in cycles:
                if train_policy in cycle:
                    unstable_opt.remove(train_policy)
                    if len(cycle) == 1:
                        stable_opt.append(train_policy)
                    else:
                        cyclic_opt.append(train_policy)
    f.write("Stable Optima: " + str(stable_opt) + "\n")
    f.write("Cyclic Optima: " + str(cyclic_opt) + "\n")
    f.write("Unstable Optima: " + str(unstable_opt) + "\n")

    f.close()


if __name__ == '__main__':
    main(sys.argv[1:])
