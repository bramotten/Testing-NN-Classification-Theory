import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

sns.set()
np.set_printoptions(threshold=256)
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 21

# if not COLAB:  # and it were not so ugly...
#     [f.name for f in matplotlib.font_manager.fontManager.ttflist]
#     plt.rcParams["font.family"] = "Charter"
#     plt.rcParams["mathtext.fontset"] = "stix"
