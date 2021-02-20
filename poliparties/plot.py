"""Plotting tools"""


from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from poliparties import analysis


def plot_training_data(ax, X, features, **kwargs):
    im = ax.imshow(X, **kwargs)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Score")
    plt.xticks(range(len(features)), features, rotation=90)
    ax.set_title("Training data")
    ax.set_xlabel("Features")
    return ax


def plot_ellipse(ax, *args, **kwargs):
    ellipse = Ellipse(*args, **kwargs)
    ax.add_patch(ellipse)
    return ax


def plot_gaussian(ax, y: np.ndarray, n_std=2, **kwargs):
    # NOTE: In higher dimensions 2 * std doesn't correspond
    # to 95% confidence interval but less
    (mean, cov) = analysis.estimate_gaussian(y)
    (w2, v) = np.linalg.eigh(cov)
    w = np.sqrt(w2)
    # rotation angle between first principal axis and x-axis
    rot = np.arccos(np.dot([1, 0], v[:, 1])) * 360. / np.pi / 2.
    ax = plot_ellipse(
        ax,
        xy=mean,
        width=2*w[1]*n_std,
        height=2*w[0]*n_std,
        angle=rot,
        **kwargs
    )
    return ax


def plot_classes2d(
        ax,
        y: np.ndarray,
        labels: np.ndarray,
        cm=plt.cm.gist_ncar,
        n_classes=20,
        **kwargs
):
    label_counts = pd.Series(labels).value_counts()
    colors = cm(np.linspace(0, 1, n_classes))
    for (label, color) in zip(label_counts.index[:n_classes], colors):
        y_ = y[labels == label, :]
        ax.scatter(*y_.T, color=color)
        if len(y) > 1:
            ax = plot_gaussian(ax, y_, color=color, n_std=2, **kwargs)
        else:
            ax.plot(*y_, color=color, marker="+")
    return ax


def plot_circles(ax, y, labels, scale_size=50, **kwargs):
    spheres = analysis.estimate_spheres(y, labels)
    label_counts = pd.Series(labels).value_counts()
    # TODO: NaN to different marker
    im = ax.scatter(
        spheres["mean_x"],
        spheres["mean_y"],
        s=scale_size*spheres["r_std"],
        c=label_counts.values,
        **kwargs
    )
    plt.colorbar(im, ax=ax)
    return ax
