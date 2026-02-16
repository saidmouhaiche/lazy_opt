from candlebark.enginecomponent.fuel import Fuel
from candlebark.enginecomponent.fuel.grain_geometry import Cruciform, UniPortGeometry
from candlebark.enginecomponent.fuel.propellants import get_ABS
from candlebark.enginecomponent.tank import Tank
from candlebark.enginecomponent.tank.pressurant import Pressurant
from candlebark.enginecomponent.tank.n2o import N2O
from candlebark.enginecomponent.chamber import Chamber
from candlebark.enginecomponent.engine import HybridRocketEngine
from candlebark.enginecomponent.feedline import Feedline
from candlebark.enginecomponent.injector import Injector
from candlebark.enginecomponent.nozzle import Nozzle
from candlebark.simulation.sim_manager import SimProcess
from candlebark.simulation.sim_status import SimMode
from candlebark.data.input_config_handler import InputConfigHandler
from candlebark.data.constants import GAS_CONSTANT as R
import pathlib

from orhelper_enums import OrLogLevel, FlightDataType
from CoolProp.CoolProp import PropsSI
import orhelper
import subprocess
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

from shapely.geometry import Point, Polygon
from shapely.affinity import rotate
from shapely.ops import unary_union
from typing import cast

from enum import Enum, auto
import math
import json
import tqdm
import shutil

class DesignVariables(Enum):
    IS_HYBRID = auto()

    # Nose cone
    NOSE_CONE_LENGTH = auto()
    NOSE_PROFILE = auto()

    # Fin
    FIN_HEIGHT = auto()
    FIN_SWEEP = auto()
    FIN_POSITION = auto()
    FIN_ROOT_CHORD = auto()
    FIN_CHORD_RATIO = auto()
    FIN_THICKNESS = auto()

    # Boattail
    BOATTAIL_PROFILE = auto()

    # Injector / nozzle
    INJECTOR_AREA = auto()
    NOZZLE_DIAMETER = auto()

    # Tank
    TANK_LENGTH = auto()
    TANK_ULLAGE = auto()
    STRUT_LENGTH = auto()

    # Grain geometry
    GRAIN_OD = auto()
    GRAIN_LENGTH = auto()
    GRAIN_N_ARMS = auto()
    GRAIN_SLOT_LENGTH = auto()
    GRAIN_SLOT_WIDTH_TIP = auto()
    GRAIN_SMOOTHNESS = auto()
    GRAIN_HUB_DIAMETER = auto()
    GRAIN_ID_MARGIN = auto()
    GRAIN_ID_RATIO = auto()
    GRAIN_INFILL = auto()  # P

    @classmethod
    def get_names(cls):
        return [dv.name for dv in cls]

    @classmethod
    def get_values(cls):
        return [dv.value for dv in cls]


# === Helper Functions ===
def load_design_defaults(path="design_defaults.json") -> dict[DesignVariables, float]:
    with open(path, "r") as f:
        raw_defaults = json.load(f)
    defaults = {}
    for key, value in raw_defaults.items():
        try:
            enum_key = DesignVariables[key]
            defaults[enum_key] = value
        except KeyError:
            print(f"[WARN] Unknown design variable in JSON: {key}")
    return defaults


def calculate_tank_mass(tank_length: float) -> float:
    density_6061_t6 = 2700
    od = math.pi * 0.0762 ** 2
    id = math.pi * 0.07 ** 2
    cross_sectional_area = od - id
    return cross_sectional_area * tank_length * density_6061_t6


def calculate_tank_volume(tank_length: float) -> float:
    radius = 0.13334 / 2
    closure_volumes = 0.0006
    return math.pi * (radius ** 2) * tank_length + closure_volumes


def calculate_oxidiser_mass(ullage: float, total_volume: float, liquid_temperature: float) -> float:
    ullage_frac = ullage / 100
    vapor_volume = ullage_frac * total_volume
    liquid_volume = (1 - ullage_frac) * total_volume

    P = PropsSI('P', 'T', liquid_temperature, 'Q', 0, 'N2O')  # Pa
    M = PropsSI('M', 'N2O') * 1000  # kg/mol
    v_liq = N2O.molar_volume_liquid(liquid_temperature)  # m³/mol

    n_liq = liquid_volume / v_liq
    n_vap = (P * vapor_volume) / (R * liquid_temperature)
    n_total = n_liq + n_vap

    return n_total * M


# === Engine Setup ===
def setup_hybrid_engine(user_config: dict[DesignVariables, float]):
    path = pathlib.Path(__file__).parent / "candlebark" / "data" / "inputs" / "monarch.toml"
    input_config_handler = InputConfigHandler(path)
    simulation = input_config_handler.get_simulation()
    monarch = simulation.eng

    def update_tank():
        liquid_temperature = 293.15
        tank_length = user_config.get(DesignVariables.TANK_LENGTH)
        ullage = user_config.get(DesignVariables.TANK_ULLAGE)

        projected_mass = calculate_tank_mass(tank_length)
        projected_volume = calculate_tank_volume(tank_length)
        projected_oxidiser_mass = calculate_oxidiser_mass(ullage, projected_volume, liquid_temperature)
        n2o_updated = N2O(projected_oxidiser_mass)
        helium = Pressurant(0.0, "HE")
        monarch.tank = Tank(n2o_updated, helium, liquid_temperature, projected_volume, projected_mass, 0.1334,
                            tank_length)

    def update_injector():
        area = user_config.get(DesignVariables.INJECTOR_AREA)
        monarch.injector.AREA = area

    def update_nozzle():
        diameter = user_config.get(DesignVariables.NOZZLE_DIAMETER)
        new_nozzle = Nozzle(0.09795, 15, 0.983, 3, diameter, 1.3)
        monarch.nozzle = new_nozzle

    def update_fuel_grain():
        od = user_config.get(DesignVariables.GRAIN_OD)
        length = user_config.get(DesignVariables.GRAIN_LENGTH)
        n_arms = user_config.get(DesignVariables.GRAIN_N_ARMS)
        slot_length = user_config.get(DesignVariables.GRAIN_SLOT_LENGTH)
        slot_width_tip = user_config.get(DesignVariables.GRAIN_SLOT_WIDTH_TIP)
        smoothness = user_config.get(DesignVariables.GRAIN_SMOOTHNESS)
        hub_diameter = user_config.get(DesignVariables.GRAIN_HUB_DIAMETER)
        infill_density = user_config.get(DesignVariables.GRAIN_INFILL)

        cruciform = Cruciform(
            hub_diameter,
            slot_length,
            slot_width_tip,
            smoothness,
            n_arms
        )
        monarch.chamber.fuel = Fuel(
            length,
            od,
            cruciform,
            get_ABS(infill_density),
        )

    update_map = {
        DesignVariables.GRAIN_HUB_DIAMETER: update_fuel_grain,
        DesignVariables.GRAIN_SLOT_LENGTH: update_fuel_grain,
        DesignVariables.GRAIN_SLOT_WIDTH_TIP: update_fuel_grain,
        DesignVariables.GRAIN_SMOOTHNESS: update_fuel_grain,
        DesignVariables.GRAIN_N_ARMS: update_fuel_grain,
        DesignVariables.TANK_LENGTH: update_tank,
        DesignVariables.TANK_ULLAGE: update_tank,
        DesignVariables.INJECTOR_AREA: update_injector,
        DesignVariables.NOZZLE_DIAMETER: update_nozzle
    }

    for key in user_config:
        if key in update_map:
            update_map[key]()

    return simulation


def run_openrocket_sim(config: dict[DesignVariables, float]):
    args = []
    for key, value in config.items():
        args.append(f"{key.name}={value}")
    import sys
    subprocess.run([sys.executable, "run_openrocket.py", *args], check=True, stdout=subprocess.DEVNULL)

def run_solver(user_overrides: dict[DesignVariables, float], verobse=True):
    defaults = load_design_defaults()
    merged_config = defaults.copy()
    merged_config.update(user_overrides)

    if verobse:
        print("Running solver with design variables:\n")
        for var, value in merged_config.items():
            print(f"{var.name} = {value}")

    if merged_config[DesignVariables.IS_HYBRID]:
        if verobse:
            print("Hybrid engine configuration detected.")
        simulation = setup_hybrid_engine(merged_config)
        # simulation.eng.chamber.fuel.plot_fuel_grain()
        engdatabranch = simulation.run(SimProcess(SimMode.HOT_FIRE))
        engdatabranch.export_rse('monarch.rse')

    # run openrocket sim
    run_openrocket_sim(merged_config)



import orhelper
from random import gauss

import numpy as np
from numpy import deg2rad
from numpy import array, empty, random, linspace, meshgrid, zeros, reshape, hstack, count_nonzero, logical_not, \
    count_nonzero, argmax, vstack
from numpy.random import beta
from numpy import array, empty, random, linspace, meshgrid, zeros, reshape, hstack

import math
from math import exp

from scipy.stats import qmc

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


class sim_batch(list):

    def __init__(self):
        # # ork name
        # super().__init__()
        # self.ork_name = 'simple.ork'

        # results
        self.stab_off_rod = []
        self.apogees = []
        self.gt_acc = []
        self.acc = []
        self.regret = []
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


    def seeding(self, x, simulate=1):
        self.x_seed = x
        if simulate:
            user_config = {
                DesignVariables.IS_HYBRID: 1,
                DesignVariables.NOSE_PROFILE: x[0, 0],
                DesignVariables.NOSE_CONE_LENGTH: x[0, 1],
                DesignVariables.FIN_HEIGHT: x[0, 2],
                DesignVariables.FIN_SWEEP: x[0, 3],
                DesignVariables.FIN_POSITION: x[0, 4],
                DesignVariables.FIN_CHORD_RATIO: x[0, 5],
                DesignVariables.BOATTAIL_PROFILE: x[0, 6],
                DesignVariables.NOZZLE_DIAMETER: x[0, 7],
                DesignVariables.INJECTOR_AREA: x[0, 8],
                DesignVariables.GRAIN_LENGTH: x[0, 9],
                DesignVariables.GRAIN_INFILL: x[0, 10]
            }
            ######################################

            print("🚀 Running solver...")
            try:
                # === RUN SIMULATION ===
                run_solver(user_config, verobse=False)

                # === ANALYZE OUTPUT ===
                df = pd.read_csv("rocket_flight_data.csv")
                max_altitude = df['FlightDataType.TYPE_ALTITUDE'].max() * 3.281  # ft
                df_trimmed = df[df['FlightDataType.TYPE_ALTITUDE'] <= 10]
                min_stability = df_trimmed['FlightDataType.TYPE_STABILITY'].min()

                feasible = int(not (min_stability >= 0 and max_altitude >= 10000))

                self.apogees.append(max_altitude)
                self.stab_off_rod.append(min_stability)
                self.physically_feasible.append(1)  # ✅ physically feasible

                # === OUTPUT ===
                print("\n=== Simulation Results Seeding ===")
                print(f"Apogee         : {max_altitude:.2f} ft")
                print(f"Min Stability  : {min_stability:.2f}")
                print(f"Feasible       : {'✅' if not feasible else '❌'}")

            except Exception as e:
                print(f"❌ Seeding failed due to unfeasible design: {e}")
                feasible = 1  # unfeasible
                self.apogees.append(None)
                self.stab_off_rod.append(None)
                self.physically_feasible.append(0)  # ❌ physically unfeasible
        else:
            feasible = 0

        self.f_seed = np.array([[feasible]])
        return

    def set_bounds(self, bounds=[0, 60, 0.13, 0.34,0,-0.06,0.3,1]):
        # bounds = [50, 60,  # fin sweep
        #           0.2, 0.3,  # fin height
        #           0, -0.06,  # x_le fin
        #           0.3, 1]  # nose cone length
        self.bounds = bounds
        return

    def run_sbao(self, hyper_params):

        def plot_latent(x, f, xxx, x_, xxx_):
            from sklearn.manifold import TSNE
            from sklearn.neighbors import KNeighborsClassifier
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            import numpy as np

            # Determine feasibility
            feas = np.array([int(fi < 1e-3) for fi in f])

            # Stack real and sample points together
            x_xxx_ = np.vstack([x_, xxx_])

            # Reduce to 2D latent space
            x_xxx_latent = TSNE(n_components=2).fit_transform(x_xxx_)

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

            # Colours based on apogees
            apogees = np.array(self.apogees)
            colors = cm.viridis(apogees / np.max(apogees))  # scale to [0, 1]

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
            norm = mcolors.Normalize(vmin=np.min(apogees), vmax=np.max(apogees))
            sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Apogee (ft)")

            ax.set_xlabel("Latent x1")
            ax.set_ylabel("Latent x2")
            ax.set_title("Latent Design Space with KNN Boundary")
            ax.grid(True, zorder=1)

            plt.savefig("live_plot.png")
            print('latent plot generated')
            plt.close()

            return x__, xxx__, f_hat_latent

        def function_call(input_row):
            # === CONFIGURATION ===

            user_config = {
                DesignVariables.IS_HYBRID: 1,
                DesignVariables.NOSE_PROFILE: input_row[0, 0],
                DesignVariables.NOSE_CONE_LENGTH: input_row[0, 1],
                DesignVariables.FIN_HEIGHT: input_row[0, 2],
                DesignVariables.FIN_SWEEP: input_row[0, 3],
                DesignVariables.FIN_POSITION: input_row[0, 4],
                DesignVariables.FIN_CHORD_RATIO: input_row[0, 5],
                DesignVariables.BOATTAIL_PROFILE: input_row[0, 6],
                DesignVariables.NOZZLE_DIAMETER: input_row[0, 7],
                DesignVariables.INJECTOR_AREA: input_row[0, 8],
                DesignVariables.GRAIN_LENGTH: input_row[0, 9],
                DesignVariables.GRAIN_INFILL: input_row[0, 10]
            }
            ######################################

            try:
                # === RUN SIMULATION ===
                run_solver(user_config, verobse=False)

                # === ANALYZE OUTPUT ===
                df = pd.read_csv("rocket_flight_data.csv")
                max_altitude = df['FlightDataType.TYPE_ALTITUDE'].max() * 3.281  # ft
                df_trimmed = df[df['FlightDataType.TYPE_ALTITUDE'] <= 10]
                min_stability = df_trimmed['FlightDataType.TYPE_STABILITY'].min()

                feasible = int(not (min_stability >= 0 and max_altitude >= 10000))

                self.apogees.append(max_altitude)
                self.stab_off_rod.append(min_stability)
                self.physically_feasible.append(1)  # ✅ physically feasible

                # === OUTPUT ===
                print("\n=== Simulation Results ===")
                print(f"Apogee         : {max_altitude:.2f} ft")
                print(f"Min Stability  : {min_stability:.2f}")
                print(f"Feasible       : {'✅' if not feasible else '❌'}")

                return feasible

            except Exception as e:
                print(f"❌ Simulation failed due to unfeasible design: {e}")
                self.apogees.append(None)
                self.stab_off_rod.append(None)
                self.physically_feasible.append(0)  # ❌ physically unfeasible
                return 1  # unfeasible

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
                f[i, :] = function_call(array([x[i, :]]))
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

        def testing_lite(x, f, f_hat):
            class_labels = f_hat.predict(x)
            confmat = confusion_matrix(f.ravel(), class_labels)
            return confmat

        def compute_metrics(confmat):
            # acc = trace(confmat) / sum(confmat, "all");
            acc = confmat.trace() / confmat.sum()
            return acc

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

        # bounds = [lower_x, upper_x, lower_y, upper_y]

        # bounds = [50, 60,   #fin sweep
        #           0.2, 0.3, # fin height
        #           0,-0.06,  # x_le fin
        #           0.3, 1]   # nose cone length

        bounds = self.bounds

        x_seed = self.x_seed
        f_seed = self.f_seed

        xxx, xxx_ = grid_discretize(res, bounds)
        # xxx = grid_search(res, bounds)
        x_sample, f_sample = sampling(number_of_samples, bounds)

        x = vstack((x_seed, x_sample))
        f = vstack((f_seed, f_sample))

        print(count_nonzero(f))

        x_ = normalise_inputs(x)

        iter = 0
        # acc = empty([iter_max + 1])
        import time
        start = time.time()
        while 1:
            data_matrix = hstack((x, f))

            # normalise x vec


            f_hat = create_surrogate(x_, f, surr)
            f_hat_pred = predict(f_hat, xxx_)

            # plot every time, inside the loop.
            # plot_latent(x, f, xxx,x_,xxx_)

            # # calculate accuracy using k-folds:
            # kf = KFold(n_splits=k, shuffle=True)
            # acc_ = np.empty(k)
            # for i_kf, (train_idx, test_idx) in enumerate(kf.split(data_matrix)):
            #     train_data = data_matrix[train_idx, :]
            #     test_data = data_matrix[test_idx, :]
            #
            #     f_hat_ = create_surrogate(train_data[:, :dims], train_data[:, dims], surr)
            #     confmat = testing_lite(test_data[:, :dims], test_data[:, dims], f_hat_)
            #     acc_[i_kf] = compute_metrics(confmat)
            #
            # acc[iter] = acc_.mean()

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
            f_star = function_call(x_star.reshape(1, -1))
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


    def print_or_stats(self):
        print(f'Apogees    {self.apogees}')
        print(f'Min Stabs  {self.stab_off_rod}')

    def print_sbao_stats(self):
        print(f'gt acc    {self.gt_acc}')
        # print(f'acc  {self.acc}')
        print(f'regret  {self.regret}')

    def get_data(self):
        # x, x_, x__, xxx, xxx_, xxx__, f, f_hat, f_hat_pred, f_hat_latent, apogees
        # return self.xxx, self.surrogate_pred, self.true_pred, self.x, self.f
        return self.x, self.x_, self.x__, self.xxx, self.xxx_, self.xxx__, self.f, self.f_hat, self.f_hat_pred, self.f_hat_latent, self.apogees, self.physically_feasible

    def save_results_to_mat(self, filename="simulation_results.mat"):
        from scipy.io import savemat
        data_dict = {
            'xxx': self.xxx,
            'f_hat_pred': self.surrogate_pred,
            'fff': self.true_pred,
            'x': self.x,
            'f': self.f,
            'gt_acc': self.gt_acc,
            # 'acc': self.acc,
            'regret': self.regret,
            'apogees': self.apogees,
            'stab_off_rod': self.stab_off_rod,
        }
        savemat(filename, data_dict)
        print(f"✅ Saved results to {filename}")

    import csv
    import os

    def save_results_to_csv(self, filename="simulation_results.csv"):
        x = np.array(self.x)
        f = np.array(self.f)
        apogees = np.array(self.apogees)

        if not (len(x) == len(f) == len(apogees)):
            print("❌ Data arrays are not the same length. Cannot export.")
            return

        data = {
            'x_0': x[:, 0],
            'x_1': x[:, 1],
            'f': f,
            'apogee': apogees
        }

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"✅ Saved results to {filename}")


if __name__ == '__main__':
    # surr = hyper_params[0]
    # epsilon = hyper_params[1]
    # number_of_samples = hyper_params[2]
    # iter_max = hyper_params[3]
    # res = hyper_params[4]
    # k = hyper_params[5]
    # dims = hyper_params[6]
    #
    # # bounds = [lower_x, upper_x, lower_y, upper_y]
    bounds = [0,0.3,      # nose profile
              0.3, 1,       # nose cone length
              0.1, 0.4,     # fin height
              0, 60,        # fin sweep
              -0.06, 0,     # fin position
              0.2, 0.5,     # fin chord ratio
              0, 0.3,     # boatail profile
              0.03, 0.04,   # nozzle diamaeter
              4.6e-5,10.5e-5, # injector area
              0.4,0.6,      # grain length
              75,95]        # grain infill

    # sweep58, height0.25
    seed = np.array([[0,        # nose profile
                      0.858,    # nose cone length
                      0.17,     # fin height
                      43.36,    # fin sweep
                      -0.03,    # fin position
                      0.357,    # fin chord ratio
                      0,        # boatail profile
                      0.03,     # nozzle diamaeter
                      7e-5,     # injector area
                      0.475,    # grain length
                      90]])     # grain infill

    # hyper_params = [surr, epsilon, number_of_samples, iter_max, res, k-folds, dimentions, liveplot boolean]
    res = 10000000
    hyper_params = ['KNN', 1, 1, 1, res, 1, 11,0]
    # hyper_params = ['KNN', 3, 50, 250, 250, 10]
    openrocket = sim_batch()
    openrocket.seeding(seed)
    openrocket.set_bounds(bounds)
    openrocket.run_sbao(hyper_params)
    openrocket.print_or_stats()
    openrocket.print_sbao_stats()
    x, x_, x__, xxx, xxx_, xxx__, f, f_hat, f_hat_pred, f_hat_latent, apogees, physically_feasible = openrocket.get_data()

    print(x, x_, x__, xxx, xxx_, xxx__, f, f_hat, f_hat_pred, f_hat_latent, apogees,physically_feasible)

    # openrocket.save_results_to_mat(filename=f'lazy_{hyper_params[6]}d_{hyper_params[2]+hyper_params[3]}runs.mat')
    # openrocket.save_results_to_csv(filename='lazy_2d_30runs.csv')
