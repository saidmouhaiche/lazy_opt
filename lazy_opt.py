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

import pandas as pd
import numpy as np
from numpy import deg2rad
from numpy import array, empty, random, linspace, meshgrid, zeros, reshape, hstack, count_nonzero, logical_not, \
    count_nonzero, argmax, vstack
from numpy.random import beta
from numpy import array, empty, random, linspace, meshgrid, zeros, reshape, hstack

import math
from math import exp

from scipy.stats import qmc

class sim_batch(list):

    def __init__(self):
        # # ork name
        # super().__init__()
        # self.ork_name = 'simple.ork'

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

        self.xxx__ = []
        self.x__ = []
        self.f_hat_latent = []

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
        self.f1.append(feasible)
        return feasible

    def seeding(self, x):
        self.x_seed = x
        feasible = self.function_call(x)
        self.f_seed = np.array([[feasible]])
        return

    def set_bounds(self, bounds):
        self.bounds = bounds
        return

    def run_sbao(self, hyper_params):

        def plot_latent(x, f, xxx, x_, xxx_):

            # give a binary mask of feasiblity, so if feas==1, then it is feasible.
            feas = np.array([int(fi[0] < 1e-3) for fi in f])

            # Stack real and sample points together
            x_xxx_ = np.vstack([x_, xxx_])

            # Reduce to 2D latent space
            # x_xxx_latent = TSNE(n_components=2).fit_transform(x_xxx_)
            x_xxx_latent = PCA(n_components=2).fit_transform(x_xxx_)

            # Split back to x and xxx projections
            x_len = x.shape[0]
            xxx_len = xxx.shape[0]
            x__ = x_xxx_latent[:x_len, :]
            xxx__ = x_xxx_latent[x_len:, :]

            # Train KNN on the latent projection
            k = 1
            f_hat_latent = KNeighborsClassifier(n_neighbors=k)
            f_hat_latent.fit(x__, f.ravel())
            f_hat_pred_latent = f_hat_latent.predict(xxx__)
            f_hat_pred_latent = f_hat_pred_latent.reshape(-1, 1)

            # Estimate resolution
            res = int(np.sqrt(xxx__.shape[0]))

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

            # Shade KNN-predicted boundary in latent space
            for i in range(xxx__.shape[0]):
                if f_hat_pred_latent[i, 0] == 0:
                    ax.scatter(
                        xxx__[i, 0], xxx__[i, 1],
                        color='green',
                        alpha=0.3,
                        s=25,
                        edgecolors='none',
                        zorder=1
                    )
                else:
                    ax.scatter(
                        xxx__[i, 0], xxx__[i, 1],
                        color='grey',
                        alpha=0.1,
                        s=25,
                        edgecolors='none',
                        zorder=1
                    )

            # Colorbar
            norm = mcolors.Normalize(vmin=np.min(f1), vmax=np.max(f1))
            sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Objective: f1")

            ax.set_xlabel("Latent x1")
            ax.set_ylabel("Latent x2")
            ax.set_title("Latent Design Space with KNN Boundary")
            ax.grid(True, zorder=1)

            plt.savefig("live_plot.png")
            print('latent plot generated')
            plt.close()

            return x__, xxx__, f_hat_latent

        def scope_bounds(input, bounds):
            # input: (N, dims) in [0, 1]
            # bounds: list of length 2*dims → [a1, b1, a2, b2, ..., ad, bd]
            dims = hyper_params[6]
            scaled_input = empty([len(input), dims])
            for i in range(dims):
                a = bounds[2 * i]
                b = bounds[2 * i + 1]
                scaled_input[:, i] = input[:, i] * (b - a) + a
            return scaled_input

        def grid_discretize(res, bounds):
            from scipy.stats import qmc
            dims = hyper_params[6]

            # Total number of samples = res^dims
            # num_points = res ** dims
            num_points = res

            # Create uniform grid in [0,1]^dims using Sobol or Latin Hypercube
            sampler = qmc.LatinHypercube(d=dims)
            xxx_ = sampler.random(n=num_points)

            # Scale to real bounds
            xxx = scope_bounds(xxx_, bounds)

            # Return normalized and scaled versions
            return xxx, xxx_

        def predict(f_hat, x):
            f_hat_pred_ = f_hat.predict(x)
            f_hat_pred = reshape(f_hat_pred_, (-1, 1))
            return f_hat_pred

        def sampling(number_of_samples, bounds):
            dims = hyper_params[6]
            sampler = qmc.LatinHypercube(d=dims)
            space_ = sampler.random(n=number_of_samples)
            space = scope_bounds(space_, bounds)

            x = empty([number_of_samples, dims])
            f = empty([number_of_samples, 1])
            for i in range(0, number_of_samples):
                x[i, :] = space[i, :]
                f[i, :] = self.function_call(array([x[i, :]]))
                print(f'sampling {i + 1}/{number_of_samples}')
            return x, f

        def create_surrogate(x, f, surr):
            f_hat = supervised_training_lite(x, f, surr)
            return f_hat

        def supervised_training_lite(x, f, model='KNN'):
            if model == 'KNN':
                k = 1
                f_hat = KNeighborsClassifier(n_neighbors=k)
                f_hat.fit(x, f.ravel())
            elif model == 'SVM':
                from sklearn import svm
                f_hat = svm.SVC(kernel='rbf')
                f_hat.fit(x, f.ravel())
            return f_hat

        def normalise_inputs(x):
            """Normalise each column of x to [0,1]."""
            mins = np.min(x, axis=0)
            maxs = np.max(x, axis=0)
            return (x - mins) / (maxs - mins + 1e-12)  # small epsilon to avoid divide-by-zero

        surr = hyper_params[0]
        epsilon = hyper_params[1]
        number_of_samples = hyper_params[2]
        iter_max = hyper_params[3]
        res = hyper_params[4]
        k = hyper_params[5]
        dims = hyper_params[6]

        # bounds = [lower_x, upper_x, lower_x1, upper_x2...]
        bounds = self.bounds

        x_seed = self.x_seed
        f_seed = self.f_seed

        xxx, xxx_ = grid_discretize(res, bounds)
        x_sample, f_sample = sampling(number_of_samples, bounds)

        x = vstack((x_seed, x_sample))
        f = vstack((f_seed, f_sample))

        print(f'\nNumber of Hits from Sampling Found: {count_nonzero(f)}')

        x_ = normalise_inputs(x)

        iter = 0
        import time
        start = time.time()
        while 1:
            f_hat = create_surrogate(x_, f, surr)
            f_hat_pred = predict(f_hat, xxx_)

            b = f_hat_pred
            a = array(logical_not(f_hat_pred), dtype='float64')
            alpha = beta(a + epsilon, b + epsilon, (len(f_hat_pred), 1))

            # Select new sample
            alpha_max_index = alpha.argmax()
            x_star = xxx[alpha_max_index, :]  # shape (dims,)
            x = np.vstack((x, x_star.reshape(1, -1)))

            # normalise x vec
            x_ = normalise_inputs(x)

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
            x__, xxx__, f_hat_latent = plot_latent(x, f, xxx, x_, xxx_)

            self.xxx__ = xxx__
            self.x__ = x__
            self.f_hat_latent = f_hat_latent


if __name__ == '__main__':
    # surr = hyper_params[0]
    # epsilon = hyper_params[1]
    # number_of_samples = hyper_params[2]
    # iter_max = hyper_params[3]
    # res = hyper_params[4]
    # k = hyper_params[5]
    # dims = hyper_params[6]
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

    # sweep58, height0.25
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

    # hyper_params = [surr, epsilon, number_of_samples, iter_max, res, k-folds, dimentions, liveplot boolean]
    res = 1000
    hyper_params = ['KNN', 1, 5, 10, res, 1, 11,1]
    # hyper_params = ['KNN', 3, 50, 250, 250, 10]
    lazy = sim_batch()
    lazy.seeding(seed)
    lazy.set_bounds(bounds)
    lazy.run_sbao(hyper_params)
