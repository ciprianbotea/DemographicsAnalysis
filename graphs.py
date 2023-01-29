import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import scatterplot
from scipy.cluster.hierarchy import dendrogram


def dendrogrm(h, instances, title, threshold):
    fig = plt.figure(title, figsize=(15, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=18, color='c')
    dendrogram(h, labels=instances, color_threshold=threshold, ax=ax)


def histogram(t, var, partition, title="Histograme"):
    fig = plt.figure(title, figsize=(14, 8))
    assert isinstance(fig, plt.Figure)
    fig.suptitle(title, fontsize=18, color='b')
    v = np.unique(partition)
    q = len(v)
    axe = fig.subplots(1, q, sharey=True)
    for i in range(q):
        axa = axe[i]
        assert isinstance(axa, plt.Axes)
        axa.set_xlabel(v[i])
        x = t[partition == v[i]][var].values
        axa.hist(x=x, rwidth=0.9, range=(t[var].min(), t[var].max()))


def instances_plot(z, partition, instances=None, title=None):
    fig = plt.figure(figsize=(10, 8))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(title, fontdict={"fontsize": 16, "color": "b"})
    ax.set_xlabel("a1", fontdict={"fontsize": 12, "color": "b"})
    ax.set_ylabel("a2", fontdict={"fontsize": 12, "color": "b"})
    ax.set_aspect(1)
    scatterplot(x=z[:, 0], y=z[:, 1], hue=partition, ax=ax)
    if instances is not None:
        n = len(instances)
        for i in range(n):
            ax.text(z[i, 0], z[i, 1], instances[i])


def map(shp, linkage_field, t, title="Harta - "):
    map_variables = list(t)
    shp1 = pd.merge(shp, t, left_on=linkage_field, right_index=True)
    for v in map_variables:
        f = plt.figure(title + v, figsize=(10, 7))
        ax = f.add_subplot(1, 1, 1)
        ax.set_title(title + v)
        shp1.plot(v, cmap="rainbow", ax=ax, legend=True)


def display():
    plt.show()
