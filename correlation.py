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
#                            CORRELATION PLOT                                 #
# --------------------------------------------------------------------------- #
def corrplot(df):

    sns.set(style="white")
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, ax=ax,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return(ax)

# ============================================================================ #
#                                CORRELATION                                   #
# ============================================================================ #
def correlation(df):
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

# ============================================================================ #
#                                ASSOCIATION                                   #
# ============================================================================ #
def association(df):
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


def assoctable(df):
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
    tests['Strength'] = np.where(tests["Cramer's V"].abs() <.1, 'Very Weak',
                        np.where(tests["Cramer's V"].abs() <.2, 'Weak',
                        np.where(tests["Cramer's V"].abs() <.3, 'Moderate',
                        'Strong')))
    return(tests)


# %%
# ---------------------------------------------------------------------------- #
#                                CORRTABLE                                     #
# ---------------------------------------------------------------------------- #
def corrtable(df, x=None, y=None, target=None, threshold=0):
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
        for pair in list(itertools.product(x,y)) :
            r = stats.pearsonr(df[pair[0]], df[pair[1]])
            tests.append(OrderedDict(
                {'x': pair[0], 'y': pair[1], "Correlation": r[0], "p-value": r[1]}))
        tests = pd.DataFrame(tests)
        tests['AbsCorr'] = tests['Correlation'].abs()
        tests['Strength'] = np.where(tests["AbsCorr"] <.25, 'Extremely Weak',
                            np.where(tests["AbsCorr"] <.35, 'Weak',
                            np.where(tests["AbsCorr"] <.4, 'Moderate',
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
            tests['Strength'] = np.where(tests["AbsCorr"] <.25, 'Extremely Weak',
                                np.where(tests["AbsCorr"] <.35, 'Weak',
                                np.where(tests["AbsCorr"] <.4, 'Moderate',
                                'Strong')))
            top = tests.loc[tests['AbsCorr'] > threshold]
            return top

# --------------------------------------------------------------------------- #
#                              AUC PLOT                                       #
# --------------------------------------------------------------------------- #
def plot_AUC(x, y1, y2, xlab, y1lab, y2lab):
   
    line1, = plt.plot(x, y1, 'b', label=y1lab)
    line2, = plt.plot(x, y2, 'r', label=y2lab)

    x1max = x[np.argmax(y1)]
    x2max = x[np.argmax(y2)]
    y1max = y1[np.argmax(y1)]
    y2max = y2[np.argmax(y2)]
    text1= "x={:.3f}, y={:.3f}".format(x1max, y1max)
    text2= "x={:.3f}, y={:.3f}".format(x2max, y2max)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("AUC")
    plt.xlabel(xlab)    
    plt.annotate(text1, xy=(x1max, y1max), xytext=(.94,.70), **kw)
    plt.annotate(text2, xy=(x2max, y2max), xytext=(.94,.40), **kw)
    plt.show()
