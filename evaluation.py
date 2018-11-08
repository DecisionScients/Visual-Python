# =========================================================================== #
#                              MODEL EVALUATION                               #
# =========================================================================== #
'''Modules for plotting various model evaluation metrics.'''

# %%
# --------------------------------------------------------------------------- #
#                                 LIBRARIES                                   #
# --------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    text1 = "x={:.3f}, y={:.3f}".format(x1max, y1max)
    text2 = "x={:.3f}, y={:.3f}".format(x2max, y2max)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(
        arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("AUC")
    plt.xlabel(xlab)
    plt.annotate(text1, xy=(x1max, y1max), xytext=(.94, .70), **kw)
    plt.annotate(text2, xy=(x2max, y2max), xytext=(.94, .40), **kw)
    plt.show()
