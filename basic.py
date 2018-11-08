# =========================================================================== #
#                                 BASIC                                       #
# =========================================================================== #
'''Basic plots, e.g. countplots, barplots, boxplots, scatterplots and
line plots'''

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
#                                COUNT_PLOT                                   #
# --------------------------------------------------------------------------- #
def countplot(df,xvar, hue=None, title='Count Plot', add_counts=False):

    sns.set(style="whitegrid", font_scale=1)
    sns.set_palette("GnBu_d")
    fig, ax = plt.subplots()
    sns.countplot(x = xvar, data = df, hue=hue, ax=ax).set_title(title)
    total = float(len(df[xvar]))
    if (add_counts == True):
        for p in ax.patches:
            height = p.get_height()
            text = '{:,} ({:.02f})'.format(height, height/total) 
            ax.text(p.get_x()+p.get_width()/2.,
                    height/2, text, fontsize=20, ha="center", color='white') 
    plt.tight_layout()
    return(ax)
# --------------------------------------------------------------------------- #
#                                  BOXPLOT                                    #
# --------------------------------------------------------------------------- #
def boxplot(df, xvar, yvar=None, title='Box Plot'):
    sns.set(style="whitegrid", font_scale=2)
    sns.set_palette("GnBu_d")
    fig, ax = plt.subplots()
    if (yvar == None):
        sns.boxplot(x=xvar, data=df, ax=ax).set_title(title)       
    else:
        sns.boxplot(x=xvar, y=yvar, data=df, ax=ax).set_title(title)       
    plt.tight_layout()
    return(ax)
# --------------------------------------------------------------------------- #
#                              SCATTERPLOT (SNS)                              #
# --------------------------------------------------------------------------- #
def sns_scatterplot(df, xvar, yvar, target, title):
    sns.set(style="whitegrid", font_scale=2)
    sns.set_palette("Set1")
    fig, ax = plt.subplots()
    sns.scatterplot(x=xvar, y=yvar, data=df, ax=ax, style=target,
                    hue=target).set_title(title)       
    plt.tight_layout()
    return(ax)

# --------------------------------------------------------------------------- #
#                              SCATTERPLOT (PLT)                              #
# --------------------------------------------------------------------------- #
def plt_scatterplot(df, xvar, yvar, target, title):
    ax = plt.scatter(df[xvar], df[yvar], c=target, cmap='viridis')    
    plt.tight_layout()
    return(ax)

# %%
# --------------------------------------------------------------------------- #
#                                 BARPLOT                                     #
# --------------------------------------------------------------------------- #
def bar_plot(df, xvar, yvar, title):
    sns.set(style="whitegrid", font_scale=2)
    sns.set_palette("GnBu_d")
    fig, ax = plt.subplots()
    sns.barplot(x=xvar, y=yvar, data=df, ax=ax, color='b').set_title(title)       
    plt.tight_layout()
    return(ax)

# --------------------------------------------------------------------------- #
#                                 QQ PLOT                                     #
# --------------------------------------------------------------------------- #
def qq_plot(x, title='QQ-Plot'):
    fig, ax = plt.subplots()
    ax = sm.qqplot(x)       
    plt.tight_layout()
    return(ax)

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

# --------------------------------------------------------------------------- #
#                                 HISTOGRAM                                   #
# --------------------------------------------------------------------------- #
def histogram(x, title):
    sns.set(style="whitegrid", font_scale=2)
    sns.set_palette("GnBu_d")
    fig, ax = plt.subplots()
    sns.distplot(x,bins=40, ax=ax, kde=False).set_title(title)    
    return(ax)


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
    
# --------------------------------------------------------------------------- #
#                                LINE PLOT                                    #
# --------------------------------------------------------------------------- #    
def plot_line(x,y, xlab, ylab):

    line = plt.plot(x, y, 'b')
    plt.ylabel(ylab)
    plt.xlabel(xlab)    
    plt.title(ylab + " by " + xlab)    
    xmax = x[np.argmax(y)]
    ymax = y[np.argmax(y)]
    text = "x={:.0f}, y={:.3f}".format(xmax, ymax)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90")
    kw = dict(xycoords='data',textcoords="axes fraction",
            arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    plt.annotate(text, xy=(xmax, ymax), xytext=(.94,.20), **kw)
    plt.show()
# --------------------------------------------------------------------------- #
#                    RANDOM FORESTS HYPERPARAMETER PLOT                       #
# --------------------------------------------------------------------------- #
def plot_rf_hyperparameter(df, param):

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best AUC', color=color)
    ax1.plot(df['Iteration'], df['Best AUC'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel(param, color=color)  # we already handled the x-label with ax1
    ax2.plot(df['Iteration'], df[param], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle(param + " Analysis")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

# --------------------------------------------------------------------------- #
#                                  MULTIPLOT                                  #
# --------------------------------------------------------------------------- #
def multiplot(df, title=None):
    '''
    Multiplot renders a histogram and box plot for each numeric variable
    in the provided dataframe. One plot is rendered per row.
        
    Args:
        df (pd.DataFrame): The dataframe to be analyzed
        title (str): The super title for the plot
        height (int): The height of the figure in inches
        width (int): The width of the figure in inches  
    '''

    # Set style, color and size
    sns.set(style="whitegrid", font_scale=2)
    sns.set_palette("GnBu_d") 

    width = 16
    height = 3 * len(df.columns)
    figsize = [width, height]       

    # Designates sub plots and title
    cols = df.columns
    fig, ax = plt.subplots(ncols = 6, nrows=len(cols), figsize=figsize)    

    # Renders count plots for each subplot 
    for col in range(len(cols)):        
        sns.distplot(a = df[cols[col]], ax=ax[col,0]).set_title("No Transformation")
        sns.boxplot(x = df[cols[col]], ax=ax[col,1]).set_title("No Transformation")
        sns.distplot(a = np.log(df[cols[col]]+1), ax=ax[col,2]).set_title("Log Transformation")
        sns.boxplot(x = np.log(df[cols[col]]+1), ax=ax[col,3]).set_title("Log Transformation")
        sns.distplot(a = df[cols[col]]**2, ax=ax[col,4]).set_title("Square Transformation")
        sns.boxplot(x = df[cols[col]]**2, ax=ax[col,5]).set_title("Square Transformation")
    plt.tight_layout()
    if title:
        fig.suptitle(title)
        fig.subplots_adjust(top=.9)
    return(fig)
 

# --------------------------------------------------------------------------- #
#                              MULTI-COUNTPLOT                                #
# --------------------------------------------------------------------------- #
def multi_countplot(df, nrows=None, ncols=None, width=None, height=None,
                    title=None): 
    '''Prints count plots for the categorical variables in a data frame.
    The plots can be customized using the number rows, the number columns
    of columns or both. Users may also designate the height and width
    of the figure containing the individual count plots.

    Args:
        ncols (int): The number of plots to render per row
        nrows (int): The number of rows of plots to render 
        height (int): The height of the figure in inches
        width (int): The width of the figure in inches  
    '''

    sns.set(style="whitegrid", font_scale=2)
    sns.set_palette("GnBu_d")
 
    # Sets number of rows and columns
    if all(v is None for v in [nrows, ncols]):
        nrows = len(df.columns)
        ncols = 1
    elif not nrows:
        nrows = -(-len(df.columns) // ncols)
    else:
        ncols = -(-len(df.columns) // nrows)  

    # Sets height and width 
    if not width:
        width = plt.rcParams.get('figure.figsize')[0]
    if not height:
        height = plt.rcParams.get('figure.figsize')[1] 
    figsize = [width, height]       

    # Designates sub plots
    fig, axes = plt.subplots(ncols = ncols, nrows=nrows, figsize=figsize)
    cols = df.columns

    # Renders count plots for each subplot 
    for ax, cols in zip(axes.flat, cols):
        sns.countplot(x = df[cols], ax=ax).set_title(cols)
        total = float(len(df))
        for p in ax.patches:
            height = p.get_height()
            text = '{:,} ({:.02f})'.format(height, height/total) 
            ax.text(p.get_x()+p.get_width()/2.,
                    height/2, text, fontsize=20, ha="center", color='white') 
    plt.tight_layout()
    if title:
        fig.suptitle(title)
        fig.subplots_adjust(top=.9)
    return(fig)

# --------------------------------------------------------------------------- #
#                              MULTI-HISTOGRAM                                #
# --------------------------------------------------------------------------- #
def multi_histogram(df: pd.DataFrame, nrows: int=None, ncols: int=None,
                    width: [int, float]=None, height: [int, float]=None,
                    title : str=None):  
    import pandas as pd

    warnings.filterwarnings('ignore')

    sns.set(style="whitegrid", font_scale=2)
    sns.set_palette("GnBu_d")
    sns.set_color_codes("dark")
    
    # Sets number of rows and columns
    if all(v is None for v in [nrows, ncols]):
        nrows = len(df.columns)
        ncols = 1
    elif not nrows:
        nrows = -(-len(df.columns) // ncols)
    else:
        ncols = -(-len(df.columns) // nrows)        

    if not width:
        width = plt.rcParams.get('figure.figsize')[0]
    if not height:
        height = plt.rcParams.get('figure.figsize')[1] 
    figsize = [width, height]       

    fig, ax = plt.subplots(ncols = ncols, nrows=nrows, figsize=figsize)    
    cols = df.columns

    for axis, cols in zip(axes.flat, cols):
        sns.distplot(a = df[cols], kde=True, ax=axis)
    plt.tight_layout()
    
    if title:
        fig.suptitle(title)
        fig.subplots_adjust(top=.9)
    return(fig)
#%%
# --------------------------------------------------------------------------- #
#                               MULTI-BOXPLOT                                 #
# --------------------------------------------------------------------------- #
def multi_boxplot(df, groupby=None, nrows=None, ncols=None, hue=None,
                    width=None, height=None, horizontal=True, legend=None,
                    ylim=None, title=None):  

    sns.set(style="whitegrid", font_scale=2)
    sns.set_palette("GnBu_d")
    sns.set_color_codes("dark")


    # Sets number of rows and columns
    cols = list(df.columns)
    if groupby is not None:
        cols.remove(groupby)
    if hue is not None:
        cols.remove(hue)
        
    if all(v is None for v in [nrows, ncols]):
        nrows = len(df.columns)
        ncols = 1
    elif not nrows:
        nrows = -(-len(df.columns) // ncols)
    else:
        ncols = -(-len(cols) // nrows)        

    # Set figure height and width
    if not width:
        width = plt.rcParams.get('figure.figsize')[0]
    if not height:
        height = plt.rcParams.get('figure.figsize')[1] 
    figsize = [width, height]       

    fig, ax = plt.subplots(ncols = ncols, nrows=nrows, figsize=figsize)

    # Render plots
    for axis, col in zip(ax.flat, cols):
        if horizontal:
            sns.boxplot(x = col, y = groupby, data=df, ax=axis, hue=hue,
                        notch=True)
            if hue is not None:
                handles, _ = axis.get_legend_handles_labels()
                axis.legend(handles, ["Female", "Male"])
                axis.legend(loc=legend) 
            if ylim is not None:
                axis.set(xlim=(0,ylim))
        else:
            sns.boxplot(x = groupby, y = col, data=df, ax=axis, hue=hue,
                        notch=True)
            if hue is not None:
                handles, _ = axis.get_legend_handles_labels()
                axis.legend(handles, ["Female", "Male"])
                axis.legend(loc=legend) 
            if ylim is not None:
                axis.set(ylim=(0,ylim))
    plt.tight_layout()
    
    if title:
        fig.suptitle(title)
        fig.subplots_adjust(top=.9)
    return(fig)

