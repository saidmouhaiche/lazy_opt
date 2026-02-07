import numpy as np
from lazy_opt import LazyOpt

def function_call(input_row):
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
    x10 = input_row[0, 9]
    x11 = input_row[0, 10]

    f1 = x1 + x2 ** 2 + x3 ** 3 + x4 ** 4 + x5 ** 5 + x6 ** 6 + x7 ** 7 + x8 ** 8 + x9 ** 9 + x10 ** 10 + x11 ** 11
    feasible = f1 > 0

    return feasible, (f1,)

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
                100,     # number of samples using DoE (design of experiment)
                50,     # maximum iterations
                res,    # resoultion of discritsation
                1,      # k-folds - depreciated
                11,     # number of dimentions
                1,      # boolean to draw a fast latent plot
                2]      # number of threads or how many function calls per iteration

if __name__ == '__main__':
    lazy = LazyOpt(function_call)
    lazy.seeding(seed)
    lazy.set_bounds(bounds)
    lazy.run_lazy_opt(hyper_params)
