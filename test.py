import sys

import real_stuff.evaluating as evaluating
import real_stuff.simulating as simulating
import real_stuff.training as training
from real_stuff.basic_imports import *


def print_results(results):
    pass  # TODO


if __name__ == "__main__":
    if len(sys.argv) == 2:
        situation = sys.argv[1]
        if situation.isdigit() == False:
            raise ValueError("First argument not a (situation) number.")
    else:
        situation = "1"

    results = []
    hw = [16, 16, 32, 16, 16]
    N = 4
    print(f"Starting {N} runs of situation {situation} with network {hw}.")
    for i in range(N):
        X, funcs, Y_prob = simulating.create_dataset(situation, seed=i)

        model, *test_sets = training.train_network(X, Y_prob, hidden_widths=hw)

        print(evaluating.test_loss(model, *test_sets))
        # results.append[evaluating.get_all_quantities_of_interest(model, *test_sets)]

    print(results)
