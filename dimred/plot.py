"""Plotting tools"""


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def create_colors(num, cm=plt.cm.gist_rainbow, fmt=None):
    """Create a sequence of colors

    """
    colors = cm(np.linspace(0, 1, num))

    if fmt == "hex":
        colors = list(map(mpl.colors.rgb2hex, colors))

    return colors


def plot_training_data(ax, X, features, **kwargs):
    im = ax.imshow(X, **kwargs)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Score")
    plt.xticks(range(len(features)), features, rotation=90)
    ax.set_title("Training data")
    ax.set_xlabel("Features")
    return ax


def plot_components(ax, V, features, label="Component", colors=None):
    for (i, v) in enumerate(V):
        kwargs = {} if colors is None else {"color": colors[i]}
        ax.plot(v, features, label="{} {}".format(label, i), **kwargs)
    ax.yaxis.grid(True)
    ax.legend()
    ax.set_title("Components in original dimensions")
    return ax
