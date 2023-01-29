import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from graphs import dendrogrm


class hclust():
    def __init__(self, t, variables=None, method="ward"):
        if variables is None:
            variables = list(t)
        self.x = t[variables].values
        self.method = method
        self.instances = list(t.index)
        self.h = linkage(self.x, method=self.method)

    def partition(self, title, clusters_no=None):
        p = self.h.shape[0]
        if clusters_no is None:
            k_dif_max = np.argmax(self.h[1:, 2] - self.h[:(p - 1), 2])
            clusters_no = p - k_dif_max
        else:
            k_dif_max = p - clusters_no
        threshold = (self.h[k_dif_max, 2] + self.h[k_dif_max + 1, 2]) / 2
        dendrogrm(self.h, self.instances, title, threshold)
        n = p + 1
        c = np.arange(n)
        for i in range(n - clusters_no):
            k1 = self.h[i, 0]
            k2 = self.h[i, 1]
            c[c == k1] = n + i
            c[c == k2] = n + i
        codes = pd.Categorical(c).codes
        return np.array(["Cluster " + str(code+1) for code in codes])

