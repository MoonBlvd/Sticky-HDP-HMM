from __future__ import division
import numpy as np
from numpy.random import choice, normal, dirichlet, beta, gamma, multinomial, exponential, binomial
from scipy.cluster.vq import kmeans2
import pdb

class StickyHDPHMM:
    def __init__(self, data, alpha=1, kappa=1, gma=1,
                 nu=2, sigma_a=2, sigma_b=2, L=20, 
                 kmeans_init=False):
        """
        Fox, E. B., Sudderth, E. B., Jordan, M. I., 
        & Willsky, A. S. (2011). A sticky HDP-HMM 
        with application to speaker diarization. 
        The Annals of Applied Statistics, 1020-1056.
        
        X | Z = k ~ N(mu_k, sigma_k^2)
        mu_k ~ N(0, s^2)
        sigma_k ~ InvGamma(a, b)
        
        data is dictionary, each item corresponds to one sequence
        
        """
        
        self.L = L
        self.alpha = alpha
        self.gma = gma

        self.data = data

        self.n = len(self.data.keys()) # number of sequences

        # randomly initialize the state list, L is the number of states
        num_instances = 0

        for key, value in self.data.items():
            num_instances += len(value)
            if kmeans_init:
                self.state[key] =  kmeans2(value, L)[1]
            else:
                self.state[key] = choice(self.L, len(value))



        self.kappa = kappa * num_instances # kappa should be scaled based on size of data

        # std = np.std(self.data)
        std = 1
        self.mu = [[] for i in range(L)]
        self.sigma = [[] for i in range(L)]

        # initialize the L clusters, compute mean and std.
        for i in range(L):
            cluster = []
            for key, value in self.data.items():
                idx = np.where(self.state[key] == i)[0]

                if idx.size:
                    if len(cluster) == 0:
                        cluster = value[idx]
                    else:
                        cluster = np.vstack([cluster, value[idx]])
            self.mu[i] = np.mean(cluster, axis=0)
            self.sigma[i] = np.cov(cluster, rowvar=False) # d*d cov matrix
                
        # initialize the stick-breaking process by GEM distribution
        stickbreaking = self._gem(gma)
        self.beta = np.array([next(stickbreaking) for i in range(L)])

        # initialize the transition matrix, PI.
        self.N = np.zeros((L, L))
        for key, value in self.data.items():
            T = value.shape[0]
            for t in range(1, self.T):
                for i in range(self.n):
                    self.N[self.state[key][t-1, i], self.state[key][t, i]] += 1
        self.M = np.zeros(self.N.shape)
        self.PI = (self.N.T / (np.sum(self.N, axis=1) + 1e-7)).T
        
        # Hyperparameters
        self.nu = nu

        self.a = sigma_a
        self.b = sigma_b
        

    def _logphi(self, x, mu, sigma):
        """
        Compute log-likelihood.
        return either a scalar or a numpy array of log likelihood of all clusters
        """
        # return -((x - mu) / sigma) ** 2 / 2 - np.log(sigma)
        if type(mu) is list:
            all_likelihood = []
            for i, _mu in enumerate(mu):
                k = _mu.shape[0]
                diff = np.reshape(x-_mu, (1, k))
                loglikelihood = np.dot(np.dot(diff, sigma[i]), diff.T) - np.log(2*k*np.pi * np.linalg.det(sigma[i]))
                all_likelihood.append(loglikelihood)
            return np.array(all_likelihood)
        else:
            k = mu.shape[0]
            diff = np.reshape(x - mu, (1, k))
            return np.dot(np.dot(diff, sigma), diff.T) - np.log(2 * k * np.pi * np.linalg.det(sigma))
        
    def _gem(self, gma):
        """
        Generate the stick-breaking process with parameter gma.
        """
        prev = 1
        while True:
            beta_k = beta(1, gma) * prev
            prev -= beta_k
            yield beta_k
            
    def generator(self):
        """
        Simulate data from the sticky HDP-HMM.
        """
        self.state = [list(np.where(multinomial(1, dirichlet(self.beta), self.N))[1])]
        for i in range(1, self.T):
            self.state.append(list(np.where(multinomial(1, self.PI[i, :]))[0][0] for i in self.state[-1]))
            
        for i in range(self.T):
            self.data.append([normal(self.clusterPars[j][0], 
                              self.clusterPars[j][1]) for j in self.state[i]])

        self.state = np.array(self.state)
        self.data = np.array(self.data)
        
        
    def sampler(self):
        """
        Run blocked-Gibbs sampling
        """
        #pdb.set_trace()

        for key, value in self.data.items():
            T = value.shape[0]
            # Step 1: backwards message passing
            messages = np.zeros((T, self.L))
            messages[-1, :] = 1
            for t in range(T - 1, 0, -1):
                messages[t - 1, :] = self.PI.dot(
                    messages[t, :] * np.exp(self._logphi(value[t, :], self.mu, self.sigma)))
                messages[t - 1, :] /= np.max(messages[t - 1, :])

            # Step 2: states by MH algorithm
            for t in range(1, T):
                j = choice(self.L)  # proposal
                k = self.state[key][t]

                logprob_accept = (np.log(messages[t, k]) -
                                  np.log(messages[t, j]) +
                                  np.log(self.PI[self.state[key][t - 1], k]) -
                                  np.log(self.PI[self.state[key][t - 1], j]) +
                                  self._logphi(value[t - 1, :],
                                               self.mu[k],
                                               self.sigma[k]) -
                                  self._logphi(value[t - 1, :],
                                               self.mu[j],
                                               self.sigma[j]))
                if exponential(1) > logprob_accept:
                    print ("state update!")
                    self.state[key][t] = j
                    self.N[self.state[key][t - 1], j] += 1
                    self.N[self.state[key][t - 1], k] -= 1
        
        # Step 3: auxiliary variables
        P = np.tile(self.beta, (self.L, 1)) + self.n
        np.fill_diagonal(P, np.diag(P) + self.kappa)
        P = 1 - self.n / P
        for i in range(self.L):
            for j in range(self.L):
                self.M[i, j] = binomial(self.M[i, j], P[i, j])

        w = np.array([binomial(self.M[i, i], 1 / (1 + self.beta[i])) for i in range(self.L)])
        m_bar = np.sum(self.M, axis=0) - w

        # pdb.set_trace()
        # input("continue...")
        # Step 4: beta and parameters of clusters
        #self.beta = _gem(self.gma)
        self.beta = dirichlet(np.ones(self.L) * (self.gma / self.L ))#+ m_bar

        # Step 5: transition matrix
        self.PI =  np.tile(self.alpha * self.beta, (self.L, 1)) + self.N
        np.fill_diagonal(self.PI, np.diag(self.PI) + self.kappa)
        # pdb.set_trace()
        for i in range(self.L):
            self.PI[i, :] = dirichlet(self.PI[i, :])
            cluster = []
            for key, value in self.data.items():
                idx = np.where(self.state[key] == i)
                if cluster == []:
                    cluster = value[idx]
                else:
                    cluster = np.vstack([cluster, value[idx,:]])
            nc = cluster.shape[0]
            if nc:
                xmean = np.mean(cluster)
                self.mu[i] = xmean / (self.nu / nc + 1)
                self.sigma[i] = (2 * self.b + (nc - 1) * np.var(cluster) +
                                 nc * xmean ** 2 / (self.nu + nc)) / (2 * self.a + nc - 1)
            else:
                self.mu[i] = normal(0, np.sqrt(self.nu))
                self.sigma[i] = 1 / gamma(self.a, self.b)


        # check log likelihood
        total_loglikelihood = 0
        for key, value in self.data.items():
            T = value.shape[0]
            emis = 0
            trans = 0
            for t in range(T):
                emis += self._logphi(value[t, :], self.mu[self.state[key][t]],
                                            self.sigma[self.state[key][t]])
                if t > 0:
                    trans += np.log(self.PI[self.state[key][t - 1], self.state[key][t]])
            total_loglikelihood = emis+trans
        print ("total log likelihood of all sequence: ", total_loglikelihood)
        #print ("state list: ", self.state)

    def getPath(self, h):
        """
        Get the estimated sample path of h.
        """
        paths = np.zeros(self.data.shape[0])
        for i, mu in enumerate(self.mu):
            paths[np.where(self.state[:, h] == i)] = mu
        return paths
