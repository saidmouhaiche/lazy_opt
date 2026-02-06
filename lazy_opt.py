import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

import math
import json
import shutil
import time

import pandas as pd
import numpy as np
from numpy import deg2rad
from numpy import array, empty, random, linspace, meshgrid, zeros, reshape, hstack, count_nonzero, logical_not, \
    count_nonzero, argmax, vstack
from numpy.random import beta
from numpy import array, empty, random, linspace, meshgrid, zeros, reshape, hstack

from scipy.stats import qmc

class LazyOpt(list):
    '''
    TODO:
    1. genralise this class to pass in any 'function_call'
    2. multiprocessing
    3. saving and loading results
    4. write docs on how to use this and not break it
    5. make it work in the same format as other optimisers, like sklearn.minimize() or something

    Hyperparameters:
        hyper_params = ['KNN',  # classificaiton surrogate
                    1,      # epsilon (exploration-explolitation parameter)
                    30,     # number of samples using DoE (design of experiment)
                    50,     # maximum iterations
                    res,    # resoultion of discritsation
                    1,      # k-folds - this is depreciated
                    11,     # number of dimentions
                    1,      # boolean to draw a fast latent plot
                    1]      # number of threads or how many function calls per iteration
    '''

    def __init__(self):
        # results
        self.f1 = []
        self.surrogate_pred = []
        self.true_pred = []
        self.xxx = []
        self.x = []
        self.f = []

        # self.surrogate_pred = f_hat_pred.reshape((res,) * dims)  # only if you want to store reshaped
        self.f_hat_pred = []
        self.f_hat = []
        self.x_ = []
        self.xxx_ = []

        self.bounds = []
        self.x_seed = []
        self.f_seed = []

        self.physically_feasible = []

    def function_call(self, input_row):
        # === CONFIGURATION ===

        x1 = input_row[0, 0]
        x2 = input_row[0, 1]
        x3 = input_row[0, 2]
        x4 = input_row[0, 3]
        x5 = input_row[0, 4]
        x6 = input_row[0, 5]
        x7 = input_row[0, 6]
        x8 = input_row[0, 7]
        x9 = input_row[0, 8]
        x10= input_row[0, 9]
        x11= input_row[0, 10]

        f1 = x1 + x2**2 + x3**3 + x4**4 + x5**5 + x6**6 + x7**7 + x8**8 + x9**9 + x10**10 + x11**11
        feasible = f1>0

        # Feasbile is defined as False!, the optimisation is formulated as a minimization problem
        print("\n=== function_call ===")
        print(f"f1      :       {f1:.2f}")
        if feasible:
            print(f"feasible      :       [ ]")
        else:
            print(f"feasible      :       [X]")

        self.physically_feasible.append(feasible)
        self.f1.append(f1)
        return feasible

    def seeding(self, x):
        self.x_seed = x
        feasible = self.function_call(x)
        self.f_seed = np.array([[feasible]])
        return

    def plot_latent(self, x, f, x_, xxx_):

        # give a binary mask of feasiblity, so if feas==1, then it is feasible.
        feas = np.array([int(fi[0] < 1e-3) for fi in f])

        # Stack real and sample points together
        x_xxx_ = np.vstack([x_, xxx_])

        # Reduce to 2D latent space
        start = time.time()
        x_xxx_latent = PCA(n_components=2).fit_transform(x_xxx_)

        # Split back to x and xxx projections
        x_len = x.shape[0]
        if x_len < 30:
            raise ValueError(f"TSNE needs a minimum of 30 points, or reduce the perplexity(k-neighbors) parameter")
        x__1 = x_xxx_latent[:x_len, :]
        x__ = TSNE(n_components=2).fit_transform(x__1)
        # from MulticoreTSNE import MulticoreTSNE as TSNE
        # tsne = TSNE(n_jobs=4) # n_jobs is number of cores (4=1.2x speedup)
        # x__ = tsne(n_components=2).fit_transform(x__1)
        end = time.time()
        print(f'\nFAST latent embedding took: {end - start}s\nFor a more structured plot please use TSNE on x__1=PCA(x_xxx_) with Mahalanobis distance')

        # === Plotting ===
        fig, ax = plt.subplots()

        # Colours based on objective value.
        f1 = np.array(self.f1)
        colors = cm.viridis(f1 / np.max(f1))  # scale to [0, 1]

        # Plot each actual x point
        for xi, yi, ci, fi in zip(x__[:, 0], x__[:, 1], colors, feas):
            ax.scatter(
                xi, yi,
                color=ci,
                edgecolors='black' if fi == 1 else 'none',
                linewidths=1,
                s=60,
                zorder=2
            )

        # Colorbar
        norm = mcolors.Normalize(vmin=np.min(f1), vmax=np.max(f1))
        sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Objective: f1")

        ax.set_xlabel("Latent x1")
        ax.set_ylabel("Latent x2")
        ax.set_title("Latent Design Space with Feasbile Outlined in Black")
        ax.grid(True, zorder=1)

        plt.savefig("live_plot.png")
        print('Saved live_plot.png')
        plt.close()

    def scope_bounds(self, input, bounds):
        # input: (N, dims) in [0, 1]
        # bounds: list of length 2*dims → [a1, b1, a2, b2, ..., ad, bd]
        dims = hyper_params[6]
        scaled_input = empty([len(input), dims])
        for i in range(dims):
            a = bounds[2 * i]
            b = bounds[2 * i + 1]
            scaled_input[:, i] = input[:, i] * (b - a) + a
        return scaled_input

    def grid_discretize(self, res, bounds):
        from scipy.stats import qmc
        dims = hyper_params[6]

        # Total number of samples = res^dims
        # num_points = res ** dims
        num_points = res

        # Create uniform grid in [0,1]^dims using Sobol or Latin Hypercube
        sampler = qmc.LatinHypercube(d=dims)
        xxx_ = sampler.random(n=num_points)

        # Scale to real bounds
        xxx = self.scope_bounds(xxx_, bounds)

        # Return normalized and scaled versions
        return xxx, xxx_

    def predict(self, f_hat, x):
        f_hat_pred_ = f_hat.predict(x)
        f_hat_pred = reshape(f_hat_pred_, (-1, 1))
        return f_hat_pred

    def sampling(self, number_of_samples, bounds):
        dims = hyper_params[6]
        sampler = qmc.LatinHypercube(d=dims)
        space_ = sampler.random(n=number_of_samples)
        space = self.scope_bounds(space_, bounds)

        x = empty([number_of_samples, dims])
        f = empty([number_of_samples, 1])
        for i in range(0, number_of_samples):
            x[i, :] = space[i, :]
            f[i, :] = self.function_call(array([x[i, :]]))
            print(f'sampling {i + 1}/{number_of_samples}')
        return x, f

    def create_surrogate(self, x, f, surr):
        f_hat = self.supervised_training_lite(x, f, surr)
        return f_hat

    def supervised_training_lite(self, x, f, model='KNN'):
        if model == 'KNN':
            k = 1
            f_hat = KNeighborsClassifier(n_neighbors=k)
            f_hat.fit(x, f.ravel())
        elif model == 'SVM':
            from sklearn import svm
            f_hat = svm.SVC(kernel='rbf')
            f_hat.fit(x, f.ravel())
        else:
            raise ValueError('Classification surrogate model is not supported! only use "KNN" or "SVM"')
        return f_hat

    def normalise_inputs(self, x):
        """Normalise each column of x to [0,1]."""
        mins = np.min(x, axis=0)
        maxs = np.max(x, axis=0)
        return (x - mins) / (maxs - mins + 1e-12)  # small epsilon to avoid divide-by-zero

    def set_bounds(self, bounds):
        self.bounds = bounds
        return

    def run_lazy_opt(self, hyper_params):

        surr = hyper_params[0]
        epsilon = hyper_params[1]
        number_of_samples = hyper_params[2]
        iter_max = hyper_params[3]
        res = hyper_params[4]
        k = hyper_params[5]
        dims = hyper_params[6]
        threads = hyper_params[7]

        # bounds = [lower_x, upper_x, lower_x1, upper_x2...]
        bounds = self.bounds

        x_seed = self.x_seed
        f_seed = self.f_seed

        xxx, xxx_ = self.grid_discretize(res, bounds)
        x_sample, f_sample = self.sampling(number_of_samples, bounds)

        x = vstack((x_seed, x_sample))
        f = vstack((f_seed, f_sample))

        print(f'\nNumber of Hits from Sampling Found: {count_nonzero(f)}')

        x_ = self.normalise_inputs(x)

        iter = 0
        import time
        start = time.time()
        while 1:
            f_hat = self.create_surrogate(x_, f, surr)
            f_hat_pred = self.predict(f_hat, xxx_)

            b = f_hat_pred
            a = array(logical_not(f_hat_pred), dtype='float64')
            alpha = beta(a + epsilon, b + epsilon, (len(f_hat_pred), 1))

            # Select new sample
            alpha_max_index = alpha.argmax()
            x_star = xxx[alpha_max_index, :]  # shape (dims,)
            x = np.vstack((x, x_star.reshape(1, -1)))

            # normalise x vec
            x_ = self.normalise_inputs(x)

            # Evaluate new point
            f_star = self.function_call(x_star.reshape(1, -1))
            f = np.vstack((f, np.array(f_star).reshape(1, -1)))
            print(f'infill function call: iter {iter}')

            if iter >= iter_max:
                break
            else:
                iter += 1

            # self.surrogate_pred = f_hat_pred.reshape((res,) * dims)  # only if you want to store reshaped
            self.true_pred = f_hat_pred
            self.xxx = xxx
            self.x = x
            self.f = f

            end = time.time()
            print(f'time for iter: {end-start}s')

        # self.surrogate_pred = f_hat_pred.reshape((res,) * dims)  # only if you want to store reshaped
        self.f_hat_pred = f_hat_pred
        self.xxx = xxx
        self.x = x
        self.f = f
        # the old f_hat is trained on 1 less point that is appended at the end of the loop
        f_hat = KNeighborsClassifier(n_neighbors=k)
        f_hat.fit(x_, f.ravel())
        self.f_hat = f_hat
        self.x_ = x_
        self.xxx_ = xxx_

        # plot once at the end
        if hyper_params[7]:
            self.plot_latent(x, f, x_, xxx_)


if __name__ == '__main__':
    # surr = hyper_params[0]
    # epsilon = hyper_params[1]
    # number_of_samples = hyper_params[2]
    # iter_max = hyper_params[3]
    # res = hyper_params[4]
    # k = hyper_params[5]
    # dims = hyper_params[6]
    # threads = hyper_params[7]
    #
    # # bounds = [lower_x1, upper_x1, lower_x2, upper_x2...]
    bounds = [0,0.3,      # nose profile
              0.3, 1,       # nose cone length
              0.1, 0.4,     # fin height
              0, 0.6,        # fin sweep
              -0.6, 0,     # fin position
              0.2, 0.5,     # fin chord ratio
              0, 0.3,     # boatail profile
              0.03, 0.04,   # nozzle diamaeter
              4.6e-5,10.5e-5, # injector area
              0.4,0.6,      # grain length
              -0.75,-0.95]        # grain infill

    seed = np.array([[-1,        # nose profile
                      0,    # nose cone length
                      0,     # fin height
                      0,    # fin sweep
                      0,    # fin position
                      0,    # fin chord ratio
                      0,        # boatail profile
                      0,     # nozzle diamaeter
                      0,     # injector area
                      0,    # grain length
                      0]])     # grain infill

    # hyper_params = [surr, epsilon, number_of_samples, iter_max, res, k-folds, dimentions, liveplot boolean, number of threads]
    res = 1000
    hyper_params = ['KNN',  # classificaiton surrogate
                    1,      # epsilon (exploration-explolitation parameter)
                    30,     # number of samples using DoE (design of experiment)
                    50,     # maximum iterations
                    res,    # resoultion of discritsation
                    1,      # k-folds - depreciated
                    11,     # number of dimentions
                    1,      # boolean to draw a fast latent plot
                    1]      # number of threads or how many function calls per iteration
    lazy = LazyOpt()
    lazy.seeding(seed)
    lazy.set_bounds(bounds)
    lazy.run_lazy_opt(hyper_params)
