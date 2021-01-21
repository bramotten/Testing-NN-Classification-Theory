import sys

import real_stuff.evaluating as evaluating
import real_stuff.simulating as simulating
import real_stuff.training as training
from real_stuff.basic_imports import *

if __name__ == "__main__":
    if len(sys.argv) == 2:
        situation = sys.argv[1]
        if situation.isdigit() == False:
            raise ValueError("First argument not a (situation) number.")
    else:
        situation = "1"

    losses = []
    prob_losses = []
    hw = [16, 16, 32, 16, 16]
    print("Starting {n} runs of situation {situation} with network {hw}.")
    for i in range(4):
        X, funcs, Y_prob = simulating.create_dataset(situation, seed=i)

        model, *test_sets = training.train_network(X, Y_prob, hidden_widths=hw)

        losses = evaluating.test_loss(model, *test_sets, prints=False)
        print(losses)
        # TODO: get appropriate values out now that it's a dict
        # losses.append(l)
        # prob_losses.append(prob_vec_l)

    # print(f"One-hot loss:{round(np.mean(losses), 3)}" +
    #       f"+-{round(np.std(losses), 3)}")
    # print(f"Probability loss:{round(np.mean(prob_losses), 3)}" +
    #       f"+-{round(np.std(prob_losses), 3)}")
