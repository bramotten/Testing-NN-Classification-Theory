#%%
import os
import pickle

from itertools import chain
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import tensorflow as tf

from hyperopt import fmin, tpe, hp, Trials
from tensorflow.keras import activations, layers, optimizers, regularizers
from tensorflow.keras.layers import Dense
from scipy.optimize import minimize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # should shut TF up a bit
sns.set()
np.set_printoptions(threshold=256)
rc = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "figure.dpi": 100,
    "figure.figsize": [9, 5],
    "font.serif": ["Charter"] + plt.rcParams["font.serif"],
    # 'text.usetex': True,
    # 'text.latex.preamble': [r'\usepackage{amsmath}']
}
plt.rcParams.update(rc)

COLAB = False
LOSS_FOLDER = './newest_test_losses/'
HYPEROPT_FOLDER = './new_hyperopt/'

#%%

all_results = {}
_, _, filenames = next(os.walk(LOSS_FOLDER))
for f in filenames:
    situation, npkl = f.split('__')
    n = int(npkl[:-4])
    f_results = pickle.load(open(LOSS_FOLDER + f, "rb"))
    try:
        all_results[situation][n] = f_results
    except:
        all_results[situation] = {n: f_results}
example = 'sin(2_pi_x)+1'
c = "_"
if COLAB:
    c = "*"
    example = example.replace('_', c)  # weird
print(all_results[example][1024])
print(all_results[example][1024]['KL'])
# print(all_results[example + '.2'][1024])

#%%
def plotter(quantity, min_evals=5):
    plt.figure()
    plt.title(f'Mean and BS 95%-CI are full lines; median is dashed')
    median_list = []
    for situation in all_results.keys():
        n_list = sorted(all_results[situation].keys())
        too_few_evals = []  # _very_ ugly
        quantitities = []
        for i, n in enumerate(n_list):
            q = all_results[situation][n][quantity]
            quantitities.append(q)
            if len(q) < min_evals:
                too_few_evals.append(i)
        for index in sorted(too_few_evals, reverse=True):
            del n_list[index]
            del quantitities[index]

        n_rep = [np.repeat(n, len(quantitities[i])) for i, n in enumerate(n_list)]
        pd_df = pd.DataFrame({
            'n': list(chain.from_iterable(n_rep)),
            quantity: list(chain.from_iterable(quantitities)),
        })
        # print(pd_df)
        situation = '$\\' + situation.replace(c, ' ').replace('pi', '\pi') + '$'
        sns.lineplot(data=pd_df, x='n', y=quantity, label=f"$p^0_0(x)=$" + situation)
        
        m = [np.median(q) for q in quantitities]
        plt.plot(n_list, m, '--', color=plt.gca().lines[-1].get_color())
        median_list.append(m)

    plt.title('')
    # Nicer y axis labeling:
    if quantity == 'KL':
        quantity = "Küllback-Leibler divergence (risk) on test data"
    elif quantity == 'MSE':
        quantity = "Mean squared error on test data"
    elif quantity == 'Training LL':
        quantity = "Log-likelihood on training data"
    elif quantity == "s":
        quantity = "Number of nonzero network parameters $s$"
    elif quantity == 'Pr. max difference':
        quantity = "Maximum $|p_0 - \hat{p}|$ on test data"
#   plotter('Training LL')
#   plotter('s')
#   plotter('Pr. max difference');
    plt.ylabel(quantity)
    plt.xlabel("$n$")
    plt.ylim(ymin=0)
    plt.show()
    return median_list

median_list = plotter('KL')

#%%
n_list = [1024, 1536, 2048, 3072, 4096]
for i, sit in enumerate(list(all_results.keys())):
    y = median_list[i]

    def mse_n_y(theta):
        diff = y - (theta[0] + theta[1] * np.array(n_list) ** theta[2])
        return sum(diff ** 2)
    res = minimize(mse_n_y, [0.04, 1, -0.9])
    p = res.x[2].round(4)
    n_space = np.linspace(n_list[0], n_list[-1], 1000)
    sits = sit.replace(c, '*')
    plt.plot(n_space, res.x[0] + res.x[1] * np.array(n_space) ** res.x[2],
             label=f"{sits}'s best fit, {res.x[0].round(4)} + {res.x[1].round(4)} $n^{{{p}}}$")
    plt.plot(n_list, y, '--', label="Median KL "+ sits, color=plt.gca().lines[-1].get_color())

plt.title('')
plt.ylim(0, 0.05)
plt.xlabel("$n$")
plt.ylabel('')
plt.legend()
plt.show()

#%%
plotter('MSE')
plotter('Training LL')
plotter('s')
plotter('Pr. max difference');

# %%
