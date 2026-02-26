from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from scipy.stats import qmc
import faiss

import numpy as np
from numpy import array, empty, random, linspace, meshgrid, zeros, reshape, hstack, count_nonzero, logical_not, \
    count_nonzero, argmax, vstack
from numpy.random import beta
import time
import matplotlib
matplotlib.use('Agg')

from multiprocessing import Pool
from lazy_plot import plot_live, plot_latent
from lazy_save import save_incremental


class lazy_opt():
    '''
    TODO:
    0. allow seeding to be optitnal but return a warning
    0. an actual live plot
    0. better pritning (feasible [X] is flipped
    0. setting hyperparams
    0. warn user not to do 20 <= power <= 30
    1. genralise this class to pass in any 'function_call' and handle any number of objectives f1,f2...
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

    def __init__(self, solver_function=None,bounds=None,hyper_params=None,seed=None,options=None):
        self.options = options
        self.solver_function = solver_function
        self.bounds = self._set_bounds(bounds)
        # if self.options.get('save_to_csv')
        if self.options.get('live_plot_draw', False):
            live_plot_draw_boolean = options['live_plot_draw']
        else:
            live_plot_draw_boolean = False
        if self.options.get('number_of_processes', False):
            number_of_processes = options['number_of_processes']
        else:
            number_of_processes = 1
        self.hyper_params = [
            hyper_params['surrogate'], # surr = hyper_params[0]
            hyper_params['epsilon'], # epsilon = hyper_params[1]
            hyper_params['number_of_DOE_samples'], # number_of_samples = hyper_params[2]
            hyper_params['number_of_iterations'], # iter_max = hyper_params[3]
            hyper_params['number_of_psuedo_candidates'], # res = hyper_params[4]
            1, # k = hyper_params[5]
            int(len(self.bounds)/2), # dims = hyper_params[6]
            live_plot_draw_boolean, # plot_boolean = hyper_params[7]
            number_of_processes # threads = hyper_params[8]
        ]
        if self.options.get('save_csv_filename',False):
            self.save_csv_filename = self.options['save_csv_filename']
        else:
            self.save_csv_filename = 'lazy_opt_results.csv'

        if self.options.get('save_csv_boolean',False):
            self.save_csv_boolean = self.options['save_csv_boolean']
        else:
            self.save_csv_boolean = False

        if self.options.get('verbose', False):
            self.verbose = self.options['verbose']
        else:
            self.verbose = False

        self.seed = seed

        # results
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

        self.x_seed = []
        self.f_seed = []

        # store solver outputs
        self.objectives = []
        self.feasible = []
        self.iteration_numbers = []

        # run functions here
        self._seeding(seed)
        self._run_lazy_opt()


    def _function_call(self, input_row):
        if self.solver_function is None:
            raise ValueError("No objective function provided! Pass one to __init__")

        feasible, objectives = self.solver_function(input_row)

        # Validate outputs
        if not isinstance(feasible, (bool, np.bool_)):
            raise TypeError(
                f"Function must return (bool, tuple), but 'feasible' is {type(feasible)}."
                f"Expected: True/False"
            )

        if not isinstance(objectives, tuple):
            raise TypeError(
                f"Function must return (bool, tuple), but 'objectives' is {type(objectives)}."
                f"Expected: (f1,) for single objective or (f1, f2, ...) for multi - objective."
                f"Example: return True, (value,)"
            )

        if self.verbose == True:
            # Assume feasible=True means that the design is feasible
            print("\n=== function_call ===")
            for i, obj in enumerate(objectives, 1):
                print(f"f{i}     :       {obj:.2f}")
            if feasible:
                print(f"feasible      :       [ ]")
            else:
                print(f"feasible      :       [X]")

        return feasible, objectives

    def _seeding(self, seed):
        '''
            WARNING:
            this function was written by claude
            -----------
          Single seed, no objectives:
          seed = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # 1D array
          lazy.seeding(seed)

          Multiple seeds, no objectives:
          seeds = np.array([
              [0.1, 0.2, 0.3, 0.4, 0.5],  # seed 1
              [0.2, 0.3, 0.4, 0.5, 0.6],  # seed 2
              [0.3, 0.4, 0.5, 0.6, 0.7],  # seed 3
          ])
          lazy.seeding(seeds)

          Multiple seeds WITH objectives (as list of tuples):
          seeds = [
              ([0.1, 0.2, 0.3, 0.4, 0.5], (10.5,)),           # (x_values, objectives_tuple)
              ([0.2, 0.3, 0.4, 0.5, 0.6], (15.2,)),           # single objective
              ([0.3, 0.4, 0.5, 0.6, 0.7], (20.1, 5.3, 8.2))  # multi-objective
          ]
          lazy.seeding(seeds, assume_feasible=True)

          Multiple seeds WITH objectives (as dict):
          seeds = {
              'x': np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                            [0.2, 0.3, 0.4, 0.5, 0.6]]),
              'objectives': [(10.5,), (15.2,)],              # list of tuples
              'feasible': [True, False]                      # optional
          }
          lazy.seeding(seeds)

        ==================================
          Usage Examples:

          # Example 1: Simple seeds, will evaluate function
          seed_simple = np.array([[0.2, 0.3], [0.4, 0.5]])
          lazy.seeding(seed_simple)

          # Example 2: Seeds with known objectives (skip function
          evaluation)
          seeds_with_obj = [
              ([0.1, 0.2, 0.3], (100.5,)),      # x_values,
          (objective1,)
              ([0.2, 0.3, 0.4], (150.2,)),
              ([0.3, 0.4, 0.5], (200.8,))
          ]
          lazy.seeding(seeds_with_obj, assume_feasible=True)

          # Example 3: Dict format with feasibility info
          seeds_dict = {
              'x': np.array([[0.1, 0.2, 0.3],
                             [0.2, 0.3, 0.4],
                             [0.3, 0.4, 0.5]]),
              'objectives': [(100.5,), (150.2,), (200.8,)],
              'feasible': [True, True, False]
          }
          lazy.seeding(seeds_dict)

          # Example 4: Multi-objective
          seeds_multi = [
              ([0.1, 0.2], (10.5, 20.3, 5.1)),    # 3 objectives
              ([0.2, 0.3], (15.2, 18.7, 6.2)),
          ]
          lazy.seeding(seeds_multi, assume_feasible=True)

        '''

        if seed is None:
            print("No seed provided, will rely on DoE sampling only")
            return

        assume_feasible = self.options['assume_feasible_seeds']

        # Detect format: dict, list of tuples, or plain numpy array
        if isinstance(seed, dict):
            # Dictionary format with x, objectives, and optionally feasible
            x_seed = np.atleast_2d(seed['x'])
            n_seeds = x_seed.shape[0]
            self.x_seed = x_seed

            # Extract objectives
            objectives_provided = seed.get('objectives', None)
            feasible_provided = seed.get('feasible', None)

            if objectives_provided is not None:
                if len(objectives_provided) != n_seeds:
                    raise ValueError(f"Number of objectives ({len(objectives_provided)}) must match number of seeds ({n_seeds})")

                # Store objectives and feasibility
                for i in range(n_seeds):
                    obj = objectives_provided[i]
                    if not isinstance(obj, tuple):
                        obj = (obj,)  # Convert to tuple if single value
                    self.objectives.append(obj)

                    # Get feasibility
                    if feasible_provided is not None:
                        self.feasible.append(feasible_provided[i])
                    else:
                        # Assume feasible if not provided
                        self.feasible.append(True)
                    self.iteration_numbers.append(0)

                # Create f_seed array from feasible values
                self.f_seed = np.array(self.feasible[-n_seeds:]).reshape(-1, 1)
                print(f'Loaded {n_seeds} seeds with provided objectives')
                return

        elif isinstance(seed, list) and len(seed) > 0 and isinstance(seed[0], tuple):
            # List of tuples format: [(x_values, objectives), ...]
            n_seeds = len(seed)

            # Extract x values and objectives
            x_list = []
            for i, (x_vals, obj_vals) in enumerate(seed):
                x_list.append(x_vals)

                # Ensure objectives is a tuple
                if not isinstance(obj_vals, tuple):
                    if hasattr(obj_vals, '__iter__'):
                        obj_vals = tuple(obj_vals)
                    else:
                        obj_vals = (obj_vals,)

                self.objectives.append(obj_vals)
                self.feasible.append(assume_feasible or True)
                self.iteration_numbers.append(0)

            self.x_seed = np.array(x_list)
            self.f_seed = np.array(self.feasible[-n_seeds:]).reshape(-1, 1)
            print(f'Loaded {n_seeds} seeds with provided objectives')
            return

        else:
            # Plain numpy array format - no objectives provided
            seed = np.atleast_2d(seed)
            n_seeds = seed.shape[0]
            self.x_seed = seed

            f_seed_list = []
            if not assume_feasible:
                # Evaluate each seed to get objectives
                for i in range(n_seeds):
                    seed_i = seed[i:i+1, :]  # (1, dims)
                    feasible, obj_tuple = self._function_call(seed_i)

                    self.feasible.append(feasible)
                    self.iteration_numbers.append(0)
                    self.objectives.append(obj_tuple)
                    f_seed_list.append(feasible)
                    if self.verbose:
                        print(f'seeding {i + 1}/{n_seeds}')
            else:
                # Assume all seeds are feasible, but still need to evaluate for objectives
                for i in range(n_seeds):
                    seed_i = seed[i:i+1, :]
                    feasible, obj_tuple = self._function_call(seed_i)

                    # Override feasible to True
                    self.feasible.append(True)
                    self.iteration_numbers.append(0)
                    self.objectives.append(obj_tuple)
                    f_seed_list.append(True)
                    if self.verbose:
                        print(f'seeding {i + 1}/{n_seeds} (assumed feasible)')

            self.f_seed = np.array(f_seed_list).reshape(-1, 1)
            print(f'seeded {n_seeds}')
        return

    def _scope_bounds(self, input, bounds):
        # input: (N, dims) in [0, 1]
        # bounds: list of length 2*dims → [a1, b1, a2, b2, ..., ad, bd]
        dims = self.hyper_params[6]
        scaled_input = empty([len(input), dims])
        for i in range(dims):
            a = bounds[2 * i]
            b = bounds[2 * i + 1]
            scaled_input[:, i] = input[:, i] * (b - a) + a
        return scaled_input

    def _grid_discretize(self, res, bounds):
        dims = self.hyper_params[6]

        # Total number of samples = res^dims
        # num_points = res ** dims
        num_points = res

        # Create uniform grid in [0,1]^dims using Sobol or Latin Hypercube
        sampler = qmc.LatinHypercube(d=dims)
        xxx_ = sampler.random(n=num_points)

        # Scale to real bounds
        xxx = self._scope_bounds(xxx_, bounds)

        # Return normalized and scaled versions
        return xxx, xxx_

    def predict(self, f_hat, x):
        f_hat_pred_ = f_hat.predict(x)
        f_hat_pred = reshape(f_hat_pred_, (-1, 1))
        return f_hat_pred

    def _sampling(self, number_of_samples, bounds):
        dims = self.hyper_params[6]
        sampler = qmc.LatinHypercube(d=dims)
        space_ = sampler.random(n=number_of_samples)
        space = self._scope_bounds(space_, bounds)

        x = empty([number_of_samples, dims])
        f = empty([number_of_samples, 1])
        for i in range(0, number_of_samples):
            x[i, :] = space[i, :]
            f[i, :], f1 = self._function_call(array([x[i, :]]))
            self.feasible.append(f[i, :])
            self.iteration_numbers.append(0)
            self.objectives.append(f1)
            if self.verbose:
                print(f'sampling {i + 1}/{number_of_samples}')

        print(f'sampled {number_of_samples}')
        return x, f

    def _create_surrogate(self, x, f, surr):
        f_hat = self._supervised_training_lite(x, f, surr)
        return f_hat

    def _make_faiss_classifier(self):
        class FaissClassifier:
            def fit(self, x, y):
                x = np.array(x, dtype=np.float32)
                self.labels_ = np.array(y)
                self.index_ = faiss.IndexFlatL2(x.shape[1])
                self.index_.add(x)
                return self

            def predict(self, x):
                x = np.array(x, dtype=np.float32)
                _, indices = self.index_.search(x, k=1)
                return self.labels_[indices[:, 0]]

        return FaissClassifier()

    def _make_hnswlib_classifier(self):
        import hnswlib

        class HNSWClassifier:
            def fit(self, x, y):
                x = np.array(x, dtype=np.float32)
                self.labels_ = np.array(y)
                self.index_ = hnswlib.Index(space='l2', dim=x.shape[1])
                self.index_.init_index(max_elements=x.shape[0], ef_construction=200, M=16)
                self.index_.add_items(x, np.arange(x.shape[0]))
                self.index_.set_ef(50)
                return self

            def predict(self, x):
                x = np.array(x, dtype=np.float32)
                indices, _ = self.index_.knn_query(x, k=1)
                return self.labels_[indices[:, 0]]

        return HNSWClassifier()

    def _supervised_training_lite(self, x, f, model='KNN'):
        if model.lower() == 'knn':
            k = 1
            f_hat = KNeighborsClassifier(n_neighbors=k)
            f_hat.fit(x, f.ravel())
        elif model.lower() == 'svm':
            f_hat = svm.SVC(kernel='rbf')
            f_hat.fit(x, f.ravel())
        elif model.lower() == 'faiss':
            f_hat = self._make_faiss_classifier()
            f_hat.fit(x, f.ravel())
        elif model.lower() == 'hnswlib':
            f_hat = self._make_hnswlib_classifier()
            f_hat.fit(x, f.ravel())
        else:
            raise ValueError('Classification surrogate model is not supported! only use "KNN", "SVM", "faiss", or "hnswlib"')
        return f_hat

    def _normalise_inputs(self, x):
        # """Normalise each column of x to [0,1]."""
        # mins = np.min(x, axis=0)
        # maxs = np.max(x, axis=0)
        # return (x - mins) / (maxs - mins + 1e-12)

        """Normalise each column of x to [0,1] using the defined bounds."""
        dims = self.hyper_params[6]
        bounds = self.bounds

        x_normalized = np.empty_like(x)
        for i in range(dims):
            lower = bounds[2 * i]
            upper = bounds[2 * i + 1]
            x_normalized[:, i] = (x[:, i] - lower) / (upper - lower)

        return x_normalized
        # small epsilon to avoid divide-by-zero

    def _set_bounds(self, bounds):
        """
        Set optimization bounds in multiple formats.
        WARNING: this function is written with claude
        Usage:
        ----------
          # List of tuples - most readable!
          opt.set_bounds([(0, 1), (-5, 5), (0, 10)])

          # Dictionary - self-documenting
          opt.set_bounds({'x': (0, 1), 'y': (-5, 5), 'z': (0, 10)})

          # Flat list - original format still works
          opt.set_bounds([0, 1, -5, 5, 0, 10])

        Parameters
        ----------
        bounds : list, tuple, or dict
            Can be specified in several formats:

            1. List of tuples: [(lower1, upper1), (lower2, upper2), ...]
               Example: [(0, 1), (-5, 5), (0, 10)]

            2. Dictionary: {'param_name': (lower, upper), ...}
               Example: {'x': (0, 1), 'y': (-5, 5), 'z': (0, 10)}

            3. Flat list: [lower1, upper1, lower2, upper2, ...]
               Example: [0, 1, -5, 5, 0, 10]

        Examples
        --------
        # List of tuples (recommended for readability)
        opt.set_bounds([(0, 1), (-5, 5), (0, 10)])

        # Dictionary with parameter names
        opt.set_bounds({'x': (0, 1), 'y': (-5, 5), 'z': (0, 10)})

        # Flat list (original format)
        opt.set_bounds([0, 1, -5, 5, 0, 10])
        """

        # Case 1: Dictionary format
        if isinstance(bounds, dict):
            param_names = sorted(bounds.keys())
            bounds_list = []
            for name in param_names:
                low, high = bounds[name]
                if low >= high:
                    raise ValueError(f"Lower bound {low} must be less than upper bound {high} for '{name}'")
                bounds_list.extend([low, high])
            print(f"Bounds set for {len(param_names)} dimensions: {param_names}")
            return bounds_list

        # Case 2: List of tuples
        if isinstance(bounds, (list, tuple)) and len(bounds) > 0 and isinstance(bounds[0], (list, tuple)):
            bounds_list = []
            for i, (low, high) in enumerate(bounds):
                if low >= high:
                    raise ValueError(f"Lower bound {low} must be less than upper bound {high} at dimension {i}")
                bounds_list.extend([low, high])
            print(f"Bounds set for {len(bounds)} dimensions")
            return bounds_list

        # Case 3: Flat list (legacy format)
        if isinstance(bounds, (list, tuple)):
            if len(bounds) % 2 != 0:
                raise ValueError(f"Flat bounds list must have even length, got {len(bounds)}")
            # Validate lower < upper for each pair
            for i in range(0, len(bounds), 2):
                if bounds[i] >= bounds[i + 1]:
                    raise ValueError(
                        f"Lower bound {bounds[i]} must be less than upper bound {bounds[i + 1]} at dimension {i // 2}")
            print(f"Bounds set for {len(bounds) // 2} dimensions")
            return list(bounds)

        # Unrecognized format
        raise ValueError(
            "Unrecognized bounds format. Use:\n"
            "  - List of tuples: [(0, 1), (-5, 5)]\n"
            "  - Dictionary: {'x': (0, 1), 'y': (-5, 5)}\n"
            "  - Flat list: [0, 1, -5, 5]"
        )

    def _run_lazy_opt(self):
        surr = self.hyper_params[0]
        epsilon = self.hyper_params[1]
        number_of_samples = self.hyper_params[2]
        iter_max = self.hyper_params[3]
        res = self.hyper_params[4]
        k = self.hyper_params[5]
        dims = self.hyper_params[6]
        plot_boolean = self.hyper_params[7]
        threads = self.hyper_params[8]

        # bounds = [lower_x, upper_x, lower_x1, upper_x2...]
        bounds = self.bounds

        x_seed = self.x_seed
        f_seed = self.f_seed

        xxx, xxx_ = self._grid_discretize(res, bounds)
        x_sample, f_sample = self._sampling(number_of_samples, bounds)

        if len(x_seed) > 0:
            x = vstack((x_seed, x_sample))
            f = vstack((f_seed, f_sample))
        else:
            x = x_sample
            f = f_sample

        print(f'\nNumber of Hits from Sampling Found: {count_nonzero(f)}')

        x_ = self._normalise_inputs(x)

        # create multiprocessing pool
        pool = None
        if threads > 1:
            print("\n WARNING: Using multiprocessing with threads > 1")
            print("   make sure your script is protected with:")
            print("   if __name__ == '__main__':")
            print("       # your lazy opt code here")
            print("   otherwise you'll get infinite process spawning!\n")

            pool = Pool(processes=threads)
            print(f'Created process pool with {threads} workers')

        iter = 0

        start = time.time()
        for iter in range(iter_max + 1):
            f_hat = self._create_surrogate(x_, f, surr)
            f_hat_pred = self.predict(f_hat, xxx_)


            # f_hat_pred_ = f_hat.predict(x)
            # f_hat_pred = reshape(f_hat_pred_, (-1, 1))

            b = f_hat_pred
            a = array(logical_not(f_hat_pred), dtype='float64')
            alpha = beta(a + epsilon, b + epsilon, (len(f_hat_pred), 1))

            if threads == 1:
                # Select new sample
                alpha_max_index = alpha.argmax()
                x_star = xxx[alpha_max_index, :]  # shape (dims,)
                x = np.vstack((x, x_star.reshape(1, -1)))

                # normalise x vec
                x_ = self._normalise_inputs(x)

                # Evaluate new point
                f_star, f1 = self._function_call(x_star.reshape(1, -1))
                self.feasible.append(f_star)
                self.iteration_numbers.append(iter + 1)
                self.objectives.append(f1)

                f = np.vstack((f, np.array(f_star).reshape(1, -1)))
                if self.verbose:
                    print(f'infill function call: iter {iter}')
            else:
                top_n_indices = np.argsort(alpha.ravel())[-threads:][::-1]  # descending order
                x_stars = xxx[top_n_indices, :]  # shape (threads, dims)
                # add top 10 x_stars to x
                x = np.vstack((x, x_stars))

                # normalise x vec
                x_ = self._normalise_inputs(x)

                # Prepare inputs: list of arrays, each shape (1, dims)
                inputs = [x_stars[i:i + 1, :] for i in range(threads)]

                # Evaluate all x_star points in parallel
                results = pool.map(self._function_call, inputs)

                # Unpack results and append to lists
                # results is a list of tuples: [(f_star_0, f1_0), (f_star_1, f1_1), ...]
                f_stars = []
                for f_star, f1 in results:
                    self.feasible.append(f_star)
                    self.iteration_numbers.append(iter + 1)
                    self.objectives.append(f1)
                    f_stars.append(f_star)
                # Stack all results
                f = np.vstack((f, np.array(f_stars).reshape(-1, 1)))
                if self.verbose:
                    print(f'infill function calls: iter {iter}, evaluated {threads} points in parallel')

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
            if self.verbose:
                print(f'time for iter: {end-start}s')
            if plot_boolean:
                plot_live(x, f, x_, xxx_, self.objectives)

            if self.save_csv_boolean:
                save_incremental(self, self.save_csv_filename, iteration=iter)

        # self.surrogate_pred = f_hat_pred.reshape((res,) * dims)  # only if you want to store reshaped
        self.f_hat_pred = f_hat_pred
        self.xxx = xxx
        self.x = x
        self.f = f
        # the old f_hat is trained on 1 less point that is appended at the end of the loop
        f_hat = self._create_surrogate(x_, f, surr)
        self.f_hat = f_hat
        self.x_ = x_
        self.xxx_ = xxx_

        # plot once at the end
        if self.options.get('latent_plot_draw', False):
            plot_latent(x, f, x_, xxx_, self.objectives)
