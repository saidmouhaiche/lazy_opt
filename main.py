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

# bounds = [lower_x1, upper_x1, lower_x2, upper_x2...]
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

seed_2 = np.array([[0.2,0.3,0.1,0,-0.6,0.2,0,0.03,4.6e-5,0.4,-0.75],[0.25,0.35,0.15,0,-0.65,0.25,0.05,0.035,4.65e-5,0.45,-0.755]])

hyper_params = {'surrogate':'KNN',
                'epsilon':1,
                'number_of_DOE_samples':30,
                'number_of_iterations':10,
                'number_of_psuedo_candidates':1000
}
options = {'live_plot_draw':True,
           'live_plot_skip':5,
           'latent_plot_draw':True,
           'number_of_processes':2,
           'assume_feasible_seeds':False
}

if __name__ == '__main__':
    lazy = LazyOpt(solver_function=function_call,
                   bounds=bounds,
                   hyper_params=hyper_params,
                   seed=seed_2,
                   options=options
                   )

