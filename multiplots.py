# =========================================================================== #
#                                 VISUAL                                      #
# =========================================================================== #
'''Modules that produce multiple plots per figure.'''

# %%
# --------------------------------------------------------------------------- #
#                                 LIBRARIES                                   #
# --------------------------------------------------------------------------- #
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rc, rcParams
import seaborn as sns

import data
import association
import correlation
#%%
# --------------------------------------------------------------------------- #
#                             QUANT_QUANT                                     #
# --------------------------------------------------------------------------- #
def quant_quant(df, x, y, title=None):
    '''
    quant_quant renders grouped bar plots and mosaic plots for two categorical 
    variables.
        
    Args:
        df (pd.DataFrame): Dataframe containing data
        x (str): The name of the quantitative independent variable
        y (str): The name of the qualitatiave target variable
        title (str): The super title for the plot
    '''

    # Set style and color
    sns.set(style="whitegrid", font_scale=1)
    sns.set_palette("GnBu_d") 

    # Set figure size
    width = 12
    height = 4
    figsize = [width, height]       

    # Designates sub plots and title
    fig, ax = plt.subplots(nrows = 1, ncols=3, figsize=figsize)   

    # Obtain count and category data    
    df = df.groupby([x,y]).size().reset_index(name='counts')
    
    # Renders count plots for each subplot 
    sns.scatterplot(x=x, y=y, data=df, ax=ax[0]).set_title('Scatter Plot of ' + x + ' and ' + y)
    sns.boxplot(data=df[[x,y]], ax=ax[1]).set_title('Box Plot of ' + x + ' and ' + y)
    ax[2] = correlation.corr_plot(df[x,y]).set_title('Correlation Between ' + x + ' and ' + y)
  
    plt.tight_layout()
    if title:
        fig.suptitle(title)
        fig.subplots_adjust(top=.95)
    return(fig)
# --------------------------------------------------------------------------- #
#                              QUAL_QUAL                                      #
# --------------------------------------------------------------------------- #
def qual_qual(df, x, y, title=None):
    '''
    qual_qual renders grouped bar plots and mosaic plots for two categorical 
    variables.
        
    Args:
        df (pd.DataFrame): Dataframe containing data
        x (str): The name of the quantitative independent variable
        y (str): The name of the qualitatiave target variable
        title (str): The super title for the plot
    '''

    # Set style and color
    sns.set(style="whitegrid", font_scale=1)
    sns.set_palette("GnBu_d") 

    # Set figure size
    width = 12
    height = 4
    figsize = [width, height]       

    # Designates sub plots and title
    fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=figsize)   

    # Obtain count and category data    
    df = df.groupby([x,y]).size().reset_index(name='counts')
    
    # Renders count plots for each subplot 
    sns.barplot(x=x, y='counts', hue=y, data=df, ax=ax)
    ax.set_title('Bar Plot of ' + x + ' by ' + y)
    association.assoc_plot(df[[x,y]]).set_title('Association Between ' + x + ' and ' + y)
  
    plt.tight_layout()
    if title:
        fig.suptitle(title)
        fig.subplots_adjust(top=.95)
    return(fig)

# --------------------------------------------------------------------------- #
#                              QUANT_QUAL                                     #
# --------------------------------------------------------------------------- #
def quant_qual(df, x, y, title=None):
    '''
    Quant_cat renders boxplots and histograms for quantitative and categorical
    variables.
        
    Args:
        df (pd.DataFrame): Dataframe containing data
        x (str): The name of the quantitative variable
        y (str): The name of the qualitatiave variable
        title (str): The super title for the plot
    '''

    # Set style and color
    sns.set(style="whitegrid", font_scale=1)
    sns.set_palette("GnBu_d") 

    # Set figure size
    width = 12
    height = 3 + len(df[y].unique()) /5 
    figsize = [width, height]       

    # Split DataSet by levels of target variable
    levels = df[y].unique()
    splits = [df.loc[df[y] == level] for level in levels]

    # Designates sub plots and title
    fig, ax = plt.subplots(nrows = 1, ncols=2, figsize=figsize)   

    # Renders plots for each subplot 
    for split in splits:     
        sns.distplot(a=split[[x]], hist=False, label=split[y].unique(),
                        ax=ax[0]).set_title('Distribution of ' + x + ' by ' + y)
    ax[0].legend()
    sns.boxplot(x=x, y=y, data=df, ax=ax[1]).set_title('Box Plot of ' + x + ' by ' + y)
 
    plt.tight_layout()
    if title:
        fig.suptitle(title)
        fig.subplots_adjust(top=.95)
    return(fig)

# --------------------------------------------------------------------------- #
#                                  QUANT                                      #
# --------------------------------------------------------------------------- #
def quant(x, log=False, square=False, sqrt=False,
               inverse=False, title=None):
    '''
    Quant renders a histogram, box plot, and qqplots for each numeric 
    variable in the provided dataframe. One plot is rendered per row.
        
    Args:
        x (pd.Series): The data to be analyzed
        log (bool): If yes, show log transformations
        square (bool): If yes, show square transformations
        sqrt (bool): If yes, show square root transformations
        inverse (bool): If yes, show multiplicative inverse transformations
        title (str): The super title for the plot
    '''

    # Set style and color
    sns.set(style="whitegrid", font_scale=1)
    sns.set_palette("GnBu_d") 

    # Set figure size
    width = 12
    height = 3 + 3 * sum([log, square, sqrt, inverse])
    figsize = [width, height]       

    # Set plot array
    ncols = 2
    nrows = 1 + sum([log, square, sqrt, inverse])
    
    # Designates sub plots and title
    fig, ax = plt.subplots(nrows = nrows, ncols=ncols, figsize=figsize)   

    # Renders count plots for each subplot
    if (nrows==1):     
        sns.distplot(a = x, ax=ax[0]).set_title(x.name + ' Histogram')
        sns.boxplot(x = x, ax=ax[1]).set_title(x.name + ' Box Plot')
    else:
        sns.distplot(a = x, ax=ax[0,0])
        sns.boxplot(x = x, ax=ax[0,1])
    if (log):
        sns.distplot(a = np.log(x+1), ax=ax[1,0]).set_title("Histogram Log(" + x.name +")")
        sns.boxplot(x = np.log(x+1), ax=ax[1,1]).set_title("Box Plot Log(" + x.name +")")
    if (square):
        sns.distplot(a = x**2, ax=ax[2,0]).set_title(r"Histogram (" + x.name +")$^2$")
        sns.boxplot(x = x**2, ax=ax[2,1]).set_title(r"Box Plot (" + x.name +")$^2$")
    if (sqrt):
        sns.distplot(a = np.sqrt(x), ax=ax[3,0]).set_title("Histogram Sqrt(" + x.name +")")
        sns.boxplot(x = np.sqrt(x), ax=ax[3,1]).set_title("Box Plot Sqrt(" + x.name +")")
    if (inverse):
        sns.distplot(a = 1/x, ax=ax[4,0]).set_title("Histogram (1/" + x.name +")")
        sns.boxplot(x = 1/x, ax=ax[4,1]).set_title("Box Plot (1/" + x.name +")")

    plt.tight_layout()
    if title:
        fig.suptitle(title)
        fig.subplots_adjust(top=.95)
    return(fig)
 
# --------------------------------------------------------------------------- #
#                               COUNTPLOT                                     #
# --------------------------------------------------------------------------- #
def countplot(df, nrows=None, ncols=None, width=None, height=None,
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
#                                HISTOGRAM                                    #
# --------------------------------------------------------------------------- #
def histogram(df: pd.DataFrame, nrows: int=None, ncols: int=None,
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
#                                 BOXPLOT                                     #
# --------------------------------------------------------------------------- #
def boxplot(df, groupby=None, nrows=None, ncols=None, hue=None,
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

