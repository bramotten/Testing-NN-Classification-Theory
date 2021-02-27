# %%
from scipy.optimize import minimize
import os
import pickle
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
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

# %%
LOSS_FOLDER = './new_test_losses/'
loss_dict = {}
_, _, filenames = next(os.walk(LOSS_FOLDER))
for f in filenames:
    situation, npkl = f.split('__')
    n = int(npkl[:-4])
    losses = pickle.load(open(LOSS_FOLDER + f, "rb"))
    try:
        loss_dict[situation][n] = losses
    except:
        loss_dict[situation] = {n: losses}
print(loss_dict['sin(2_pi_x)+1'][1536])
print(loss_dict['sin(2_pi_x)+1.2'][1536])

# %%
min_length = 5
plt.figure()
# plt.title(f'Medians. KL lines are full, MSE lines dotted.')
median_list = []
for situation in loss_dict.keys():
    n_list = sorted(loss_dict[situation].keys())
    to_remove = []  # _very_ ugly
    kl_list = []
    mse_list = []
    for i, n in enumerate(n_list):
        kl = loss_dict[situation][n]['KL']
        kl_list.append(kl)
        mse = loss_dict[situation][n]['MSE']
        mse_list.append(mse)
        print(situation, n, len(kl))
        if len(kl) < min_length:
            to_remove.append(i)
    for index in sorted(to_remove, reverse=True):
        del n_list[index]
        del kl_list[index]
        del mse_list[index]

    n_rep = [np.repeat(n, len(kl_list[i])) for i, n in enumerate(n_list)]
    pd_df = pd.DataFrame({
        'n': list(chain.from_iterable(n_rep)),
        'KL loss': list(chain.from_iterable(kl_list)),
        'MSE loss': list(chain.from_iterable(mse_list))
    })
    # print(pd_df)
    situation = '$\\' + situation.replace('_', ' ').replace('pi', '\pi') + '$'
    sns.lineplot(data=pd_df, x='n', y='KL loss', label="KL "+situation)
    sns.lineplot(data=pd_df, x='n', y='MSE loss', label="MSE "+situation,
                 color=plt.gca().lines[-1].get_color(), marker='o')
    plt.plot(n_list, [np.median(kl_inner) for kl_inner in kl_list], '--',
             color=plt.gca().lines[-1].get_color())
    plt.plot(n_list, [np.median(mse_inner) for mse_inner in mse_list], '--',
             color=plt.gca().lines[-1].get_color(), marker='o')

    median_list.append([np.median(kl_inner) for kl_inner in kl_list])

plt.title('')
plt.ylim(0, 0.05)
plt.xlabel("$n$")
plt.ylabel('')
plt.show()


# plt.plot(n_list, [np.median(kl_inner) for kl_inner in kl_list],
#          label=f"{situation}", marker='o')
# plt.plot(n_list, [np.median(mse_inner) for mse_inner in mse_list],
#          '--', marker='x', color=plt.gca().lines[-1].get_color())
# plt.legend()
# plt.show()


# %%
for i, sit in enumerate(list(loss_dict.keys())):
    print('\n' + sit)
    y = median_list[i]
    for j in range(len(n_list)):
        print(n_list[j], y[j].round(5))

    def mse_n_y(theta):
        diff = y - (theta[0] + np.array(n_list) ** theta[1])
        return sum(diff ** 2)
    res = minimize(mse_n_y, [0.04, -0.9])
    p = res.x[1].round(4)
    n_space = np.linspace(n_list[0], n_list[-1], 1000)
    plt.plot(n_space, res.x[0] + np.array(n_space) ** res.x[1],
             label=f"{sit}'s least MSE fit, {res.x[0].round(4)} + $n^{{{p}}}$")
    plt.plot(n_list, y, '--', label="KL "+sit, color=plt.gca().lines[-1].get_color())

plt.title('')
plt.ylim(0, 0.05)
plt.xlabel("$n$")
plt.ylabel('')
plt.legend()
plt.show()

# %%
