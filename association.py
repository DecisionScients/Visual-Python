# =========================================================================== #
#                               ASSOCIATION                                   #
# =========================================================================== #
'''Modules for creating association plots between categorical variables'''

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

# ============================================================================ #
#                                ASSOCIATION                                   #
# ============================================================================ #


def assoc_plot(df):
    df = df.select_dtypes(include=['object'])
    cols = list(df.columns.values)
    corrM = np.zeros((len(cols), len(cols)))

    for col1, col2 in itertools.combinations(cols, 2):
        idx1, idx2 = cols.index(col1), cols.index(col2)
        corrM[idx1, idx2] = cramers_corrected_stat(
            pd.crosstab(df[col1], df[col2]))
        corrM[idx2, idx1] = corrM[idx1, idx2]

    assoc = pd.DataFrame(corrM, index=cols, columns=cols)

    mask = np.zeros_like(assoc, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(assoc, mask=mask, ax=ax, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=True, yticklabels=True)
    ax.set_title("Cramer's V Association between Variables")
    return(ax)

# %%
# ---------------------------------------------------------------------------- #
#                                  CRAMER'S V                                  #
# ---------------------------------------------------------------------------- #


def cramers_corrected_stat(contingency_table):
    """ calculate Cramers V statistic for categorical-categorical association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328

        Args:
            contingency_table (pd.DataFrame): Contingency table containing
                                              counts for the two variables
                                              being analyzed
        Returns:
            float: Corrected Cramer's V measure of Association                                    
    """
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2/n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


# %%
# ---------------------------------------------------------------------------- #
#                                ASSOCTABLE                                    #
# ---------------------------------------------------------------------------- #


def assoc_table(df):
    '''For a dataframe containing categorical variables, this function 
    computes a series of association tests for each pair of categorical
    variables. It returns the adjusted Cramer's V measure of 
    association between the pairs of categorical variables.  Note, this 
    is NOT  a hypothesis test. 

    Args:
        df (pd.DataFrame): Data frame containing categorical variables

    Returns:
        Data frame containing the results of the pairwise association measures.
    '''
    df = df.select_dtypes(include='object')
    terms = df.columns

    tests = []
    for pair in list(combinations(terms, 2)):
        x = df[pair[0]]
        y = df[pair[1]]
        ct = pd.crosstab(x, y)
        ct = pd.crosstab(x, y)
        cv = cramers_corrected_stat(ct)
        tests.append(OrderedDict(
            {'x': pair[0], 'y': pair[1], "Cramer's V": cv}))
    tests = pd.DataFrame(tests)
    tests['Strength'] = np.where(tests["Cramer's V"].abs() < .1, 'Very Weak',
                                 np.where(tests["Cramer's V"].abs() < .2, 'Weak',
                                          np.where(tests["Cramer's V"].abs() < .3, 'Moderate',
                                                   'Strong')))
    return(tests)
