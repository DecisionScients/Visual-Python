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
import seaborn as sns
import scipy
from scipy import stats

# ---------------------------------------------------------------------------- #
#                                CORR_PLOT                                     #
# ---------------------------------------------------------------------------- #


def corr_plot(df):

    sns.set(style="white")

    df = df.select_dtypes(include=['float64', 'int64'])
    corr = df.corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=True, yticklabels=True)
    ax.set_title("Correlation between Variables")
    return(ax)

# ---------------------------------------------------------------------------- #
#                                CORR_TABLE                                    #
# ---------------------------------------------------------------------------- #


def corr_table(df, x=None, y=None, target=None, threshold=0):
    '''For a dataframe containing numeric variables, this function 
    computes pairwise pearson's R tests of correlation correlation.

    Args:
        df (pd.DataFrame): Data frame containing numeric variables
        pairs (list): Pairs of columns to evaluate
        threshold (float): Threshold above which correlations should be
                           reported.

    Returns:
        Data frame containing the results of the pairwise tests of correlation.
    '''
    tests = []
    if x is not None:
        for pair in list(itertools.product(x, y)):
            r = stats.pearsonr(df[pair[0]], df[pair[1]])
            tests.append(OrderedDict(
                {'x': pair[0], 'y': pair[1], "Correlation": r[0], "p-value": r[1]}))
        tests = pd.DataFrame(tests)
        tests['AbsCorr'] = tests['Correlation'].abs()
        tests['Strength'] = np.where(tests["AbsCorr"] < .25, 'Extremely Weak',
                                     np.where(tests["AbsCorr"] < .35, 'Weak',
                                              np.where(tests["AbsCorr"] < .4, 'Moderate',
                                                       'Strong')))
        top = tests.loc[tests['AbsCorr'] > threshold]
        return top
    else:
        df2 = df.select_dtypes(include=['int', 'float64'])
        terms = df2.columns
        if target:
            if target not in df2.columns:
                df2 = df2.join(df[target])
            for term in terms:
                x = df2[term]
                y = df2[target]
                r = stats.pearsonr(x, y)
                tests.append(OrderedDict(
                    {'x': term, 'y': target, "Correlation": r[0], "p-value": r[1]}))
            tests = pd.DataFrame(tests)
            tests['AbsCorr'] = tests['Correlation'].abs()
            top = tests.loc[tests['AbsCorr'] > threshold]
            return top
        else:
            for pair in list(combinations(terms, 2)):
                x = df2[pair[0]]
                y = df2[pair[1]]
                r = stats.pearsonr(x, y)
                tests.append(OrderedDict(
                    {'x': pair[0], 'y': pair[1], "Correlation": r[0], "p-value": r[1]}))
            tests = pd.DataFrame(tests)
            tests['AbsCorr'] = tests['Correlation'].abs()
            tests['Strength'] = np.where(tests["AbsCorr"] < .25, 'Extremely Weak',
                                         np.where(tests["AbsCorr"] < .35, 'Weak',
                                                  np.where(tests["AbsCorr"] < .4, 'Moderate',
                                                           'Strong')))
            top = tests.loc[tests['AbsCorr'] > threshold]
            return top
