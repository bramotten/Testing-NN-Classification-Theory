import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # should shut TF up

sns.set()
np.set_printoptions(threshold=256)
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix",
      "figure.dpi": 120,
      "figure.figsize": [8, 5]
     }
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Garamond"] + plt.rcParams["font.serif"]


# if not COLAB:  # and it were not so ugly...
#     [f.name for f in matplotlib.font_manager.fontManager.ttflist]
#     plt.rcParams["font.family"] = "Charter"
#     plt.rcParams["mathtext.fontset"] = "stix"
