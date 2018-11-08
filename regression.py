# =========================================================================== #
#                                 VISUAL                                      #
# =========================================================================== #
'''Modules for creating scatterplots, histograms, barplots, and other 
visualizations.'''

# %%
# --------------------------------------------------------------------------- #
#                                 LIBRARIES                                   #
# --------------------------------------------------------------------------- #
import collections
from collections import OrderedDict
import itertools
from itertools import combinations
from itertools import product
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.pylab import rc, rcParams
import statsmodels.api as sm
import seaborn as sns
import scipy
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn import preprocessing
import statistics as stat
import tabulate
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------- #
#                               REGRESSION LINE                               #
# --------------------------------------------------------------------------- #


def regression_plot(df, xvar, yvar, title, ci=None):
    sns.set(style="whitegrid", font_scale=2)
    sns.set_palette("GnBu_d")
    fig, ax = plt.subplots()
    ax = sns.regplot(x=xvar, y=yvar, data=df, ci=ci, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return(ax)

# --------------------------------------------------------------------------- #
#                               RESIDUALS PLOT                                #
# --------------------------------------------------------------------------- #


def residuals_plot(df, xvar, yvar, title):
    sns.set(style="whitegrid", font_scale=2)
    sns.set_palette("GnBu_d")
    fig, ax = plt.subplots()
    ax = sns.residplot(x=xvar, y=yvar, data=df)
    ax.set_title(title)
    plt.tight_layout()
    return(ax)
