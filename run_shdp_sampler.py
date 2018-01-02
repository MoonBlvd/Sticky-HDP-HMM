import numpy as np
from numpy.random import choice, dirichlet, binomial, multinomial, beta, gamma, normal,exponential#, gaussian
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
import csv
from shdp import StickyHDPHMM
import seaborn as sns

def read_data(file_path):
    # The read-in data should be a N*W matrix,
    # where N is the length of the time sequences,
    # W is the number of sensors/data features
    i = 0
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        for line in reader:
            line = np.array(line, dtype = 'float') # str2float
            if i == 0:
                data = line
            else:
                data = np.vstack((data, line))
            i += 1
    return data

if __name__ == "__main__":
    data_path = "obs_data_16d.csv"
    data = read_data(data_path)
    data = data[15500:16500,0:1]

    DP = StickyHDPHMM(data, kappa=10, L=20,
                        kmeans_init=True)
    i = 0
    while i < 1000:
        print("iter: ", i)
        DP.sampler()
        i+=1

    path = DP.getPath(0)
    plt.figure(1)
    sns.heatmap(DP.state[:, 0:1].T, cbar=False)
    plt.figure(2)
    plt.plot(range(DP.T), path, 'r')
    plt.show()
    all_states, state_counts = np.unique(DP.state, return_counts=True)
    print("all states: ", all_states)
    print("state counts: ", state_counts)