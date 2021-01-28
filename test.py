import sys

import real_stuff.evaluating as evaluating
import real_stuff.simulating as simulating
import real_stuff.training as training
from real_stuff.basic_imports import *


def multiple_runs(situation, N, hw, quantile_prints=True):
    if quantile_prints:
        print(f"Starting {N} runs of situation {situation} with network {hw}.")
    for i in range(N):
        X, funcs, Y_prob = simulating.create_dataset(situation, seed=i)

        model, *test_sets = training.train_network(X, Y_prob, hidden_widths=hw)

        if i == 0:
            non_list_results = evaluating.get_all_quantities_of_interest(model, *test_sets)
            results = {k: [v] for k, v in non_list_results.items()}
        else:
            new_results = evaluating.get_all_quantities_of_interest(model, *test_sets)
            for k, v in new_results.items():
                results[k].append(v)
    if quantile_prints:
        q = .25
        quantiles = {k: np.quantile(v, [q, 1 - q]) for k, v, in results.items()}
        print(f"{q}%, {1 - q}% quantiles of results:")
        for k, v in quantiles.items():
            print(f"{k}: {[round(v_i, 4) for v_i in v]}")
    return results


if __name__ == "__main__":
    if len(sys.argv) == 2:
        situation = sys.argv[1]
        # if situation.isdigit() == False:
        #     raise ValueError("First argument not a (situation) number.")
    else:
        situation = "1"

    hw = [16 for _ in range(5)]
    N = 3
    multiple_runs(situation, N, hw)
