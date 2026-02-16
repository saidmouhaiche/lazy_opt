import numpy as np
from lazy_opt import LazyOpt

def function_call(input_row):
    # === CONFIGURATION ===

    x1 = input_row[0, 0]
    x2 = input_row[0, 1]

    f1 = (3 * (1 - x1) ** 2 * np.exp(-(x1 ** 2) - (x2 + 1) ** 2)
          - 10 * (x1 / 5 - x1 ** 3 - x2 ** 5) * np.exp(-x1 ** 2 - x2 ** 2)
          - 1 / 3 * np.exp(-(x1 + 1) ** 2 - x2 ** 2))

    feasible = f1 < 2

    # % quantization constriant (anything !!greater!! than 2 is good-stable)
    # f_star = z_values < 2; % since 0 is good, take the inverse of the constarint ie <
    return feasible, (f1,)

# bounds = [lower_x1, upper_x1, lower_x2, upper_x2...]
bounds = [-3,3,
          -3,3]

bounds = [(-3,3),
          (-3,3)]

bounds = {
    'x1':(-3,3),
    'x2':(-3,3)
}

hyper_params = {'surrogate':'KNN',
                'epsilon':1,
                'number_of_DOE_samples':30,
                'number_of_iterations':10,
                'number_of_psuedo_candidates':1000
}
options = {'live_plot_draw':True,
           'latent_plot_draw':False,
           'number_of_processes':1,
           'save_csv_filename':'save_csv.csv',
           'save_csv_boolean':True
}

if __name__ == '__main__':
    lazy = LazyOpt(solver_function=function_call,
                   bounds=bounds,
                   hyper_params=hyper_params,
                   seed=None,
                   options=options
                   )
