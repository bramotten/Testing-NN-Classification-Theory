import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

sns.set()
np.set_printoptions(threshold=256)
plt.rcParams['figure.figsize'] = [12, 7]  # ~ 80 chars wide
plt.rcParams['font.size'] = 19

[f.name for f in matplotlib.font_manager.fontManager.ttflist]
plt.rcParams["font.family"] = "Charter"
plt.rcParams["mathtext.fontset"] = "stix"
