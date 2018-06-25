import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import animation
from matplotlib.colors import PowerNorm
from cycler import cycler
import csv
import seaborn as sns

from shdp_multi_dim import StickyHDPHMM

if __name__ == '__main__':

    H = 1
    L = 10
    max_iter = 1000
    colors = ['r', 'b', 'g']
    # data = np.loadtxt("simulated_data.txt")
    # file = 'Brightness_features.csv'

    T = data.shape[0]
    vmin, vmax = np.min(data) * 0.5, np.max(data) * 1.5
    xs = np.logspace(np.log10(vmin), np.log10(vmax), 100)
    logxs = np.log10(xs)
    logdata = np.log10(data)

    hdp = StickyHDPHMM(logdata, L=L)  # , kmeans_init=True)
    shdp = StickyHDPHMM(logdata, kappa=10, L=L,
                        kmeans_init=False)

    for i in range(max_iter):
        shdp.sampler()

# def init():
#     for h in range(H):
#         line_shdp[h].set_data([], [])
#         dist_shdp[h].set_data([], [])
#         areas[h].set_xy([(0, 1), (0, 1)])
#
#     text.set_text("")
#     trans_shdp.set_data(shdp.PI)
#     for h in range(H):
#         ax4.add_patch(areas[h])
#     return line_shdp + dist_shdp + [trans_shdp, text] + areas


# def update(t):
#     shdp.sampler()
#     for h in range(H):
#         estimates_shdp = shdp.getPath(h)
#         line_shdp[h].set_data(np.arange(T), estimates_shdp)  # 10**
#         density = gaussian_kde(estimates_shdp)
#         density.set_bandwidth(0.1)
#         ys = density(logxs)
#
#         areas[h].set_xy(list(zip(ys, xs)) + [(0, xs[-1]), (0, xs[0])])
#         dist_shdp[h].set_data(ys, xs)
#
#     # sns.heatmap(shdp.state[:, 0:1].T, ax=ax2, cbar=False)
#     all_states, state_counts = np.unique(shdp.state, return_counts=True)
#     # print("all states: ", all_states)
#     # print("state counts: ", state_counts)
#     # print("PI: ", shdp.PI)
#     trans_shdp.set_data(shdp.PI.copy())
#     text.set_text("MCMC iteration {0}".format(t))
#     return line_shdp + dist_shdp + [trans_shdp, text] + areas