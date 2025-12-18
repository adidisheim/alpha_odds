# basic_parameters
# Created by Antoine Didisheim, at 06.08.19
# job: store default basic_parameters used throughout the projects in single .py

import datetime
import itertools
import time
from enum import Enum
import numpy as np
import pandas as pd
import socket
import os
import hashlib




##################
# Enum
##################

class PenalizationType(Enum):
    L1 = 'l1'
    L2 = 'l2'
    ELASTICNET = 'elasticnet'
    NONE = 'none'


class SpreadTopKCriterion(Enum):
    VANILLA = 'vanilla'
    Q100 = 'q100'



##################
# Running GRID
##################

POSSIBLE_Y_VARIABLE = [
    "delta_avg_odds",
    # "delta_back_then_lay_odds",
    # "delta_lay_then_back_odds",
    # "delta_start_limit_back_then_lay_odds",
    # "delta_start_limit_lay_then_back_odds",

    "delta_avg_odds_q_100",
    # "delta_back_then_lay_odds_q_100",
    # "delta_lay_then_back_odds_q_100",
    # "delta_start_limit_back_then_lay_odds_q_100",
    # "delta_start_limit_lay_then_back_odds_q_100",

    # "win"

    # "delta_avg_odds_q_1000",
    # "delta_back_then_lay_odds_q_1000",
    # "delta_lay_then_back_odds_q_1000",
    # "delta_start_limit_back_then_lay_odds_q_1000",
    # "delta_start_limit_lay_then_back_odds_q_1000"
]

SHARED_GRID = [
    ['grid', 'start_ins_year', [2000]],
    ['grid', 'y_var', POSSIBLE_Y_VARIABLE],
    ['grid', 'topk_restriction', [1, 2, 3]],
    ['grid', 't_definition', [1, 2, 3]],
    ['grid', 'spread_restriction', [-1, 0.1, 0.05]],
    ['grid', 'spread_top_k_criterion', [SpreadTopKCriterion.VANILLA, SpreadTopKCriterion.Q100]],
]


##################
# constant
##################

class Constant:
    if socket.gethostname() == 'UML-FNQ2JDW1GV':
        RES_DIR = f'./res/'
        DATA_DIR = f'./data/'
    else:
        RES_DIR = f'/data/projects/punim2039/alpha_odds/res/'
        DATA_DIR = f'/data/projects/punim2039/alpha_odds/data/'

##################
# params classes
##################

class DataParams:
    def __init__(self):
        self.no_data = ''

class LassoModelParams:
    def __init__(self):
        self.name_ = 'lasso'
        self.alpha = 0.1

class RandomForestModelParams:
    def __init__(self):
        self.name_ = 'random_forest'
        self.n_estimators = 100
        self.max_depth = None

class XGBoostModelParams:
    def __init__(self):
        self.name_ = 'xgboost'
        self.n_estimators = 100
        self.max_depth = 6
        self.learning_rate = 0.1

class GridParams:
    def __init__(self):
        self.oos_year = 2025
        self.start_ins_year = 2000
        self.y_var = 'delta_avg_odds'
        self.t_definition = None
        self.topk_restriction = 3


# store all basic_parameters into a single object
class Params:
    def __init__(self):
        self.name_detail = 'default'
        self.name = ''
        self.use_hash = True
        self.seed = 12345
        self.data = DataParams()
        self.grid = GridParams()
        self.model = LassoModelParams()

    def get_vec_dir_per_year(self):
        d = f'yearly_vec{os.sep}{self.dict_to_string_for_dir(self.vec.__dict__, old_style=True)}{os.sep}{self.dict_to_string_for_dir(self.data.__dict__, old_style=True)}{os.sep}{os.sep}{self.grid.year}{os.sep}'
        os.makedirs(d, exist_ok=True)
        return d

    def get_vec_dir_merged(self):
        d = f'merged_vec{os.sep}{self.dict_to_string_for_dir(self.vec.__dict__, old_style=True)}{os.sep}{self.dict_to_string_for_dir(self.data.__dict__, old_style=True)}{os.sep}'
        os.makedirs(d, exist_ok=True)
        return d

    def get_model_grid_dir(self, old_style =True):
        d = f'{Constant.RES_DIR}{os.sep}model{os.sep}{self.dict_to_string_for_dir(self.grid.__dict__, old_style=old_style)}{os.sep}{self.dict_to_string_for_dir(self.model.__dict__, old_style=old_style)}{os.sep}'
        os.makedirs(d, exist_ok=True)
        return d


    def update_model_name(self):
        n = self.name_detail

        self.name = n

    def print_values(self):
        """
        Print all basic_parameters used in the model
        """
        for key, v in self.__dict__.items():
            try:
                print('########', key, '########')
                for key2, vv in v.__dict__.items():
                    print(key2, ':', vv)
            except:
                print(v)

        print('----', flush=True)

    def update_param_grid(self, grid_list, id_comb, verbose = True):
        ind = []
        for l in grid_list:
            t = np.arange(0, len(l[2]))
            ind.append(t.tolist())
        combs = list(itertools.product(*ind))
        if verbose:
            print('comb', str(id_comb + 1), '/', str(len(combs)))
        c = combs[id_comb]

        for i, l in enumerate(grid_list):
            self.__dict__[l[0]].__dict__[l[1]] = l[2][c[i]]

    def finalize_parameters(self, verbose=True):
        np.random.seed(self.seed)
        if verbose:
            self.update_model_name()  # automatically create a unique name for the experiment results.
        # create a unique directory name with the basic_parameters of the experiment (the basic_parameters defines the name automatically)
        save_dir = f'{self.model.res_dir}{self.name}/'
        os.makedirs(save_dir, exist_ok=True)
        # print and save the final basic_parameters
        self.save(
            save_dir)  # this save the param object with current configuration. So we will never forget the basic_parameters of each experiment run.

    def save(self, save_dir, file_name='/basic_parameters.p'):
        # simple save function that allows loading of deprecated basic_parameters object
        df = pd.DataFrame(columns=['key', 'value'])

        for key, v in self.__dict__.items():
            try:
                for key2, vv in v.__dict__.items():
                    temp = pd.DataFrame(data=[str(key) + '_' + str(key2), vv], index=['key', 'value']).T
                    df = pd.concat([df, temp], axis=0)
                    # df = df.append(temp)

            except:
                temp = pd.DataFrame(data=[key, v], index=['key', 'value']).T
                df = pd.concat([df, temp], axis=0)
                # df = df.append(temp)
            df.to_pickle(save_dir + file_name, protocol=4)
        # return df

    def load(self, load_dir, file_name='/basic_parameters.p'):
        # simple load function that allows loading of deprecated basic_parameters object
        df = pd.read_pickle(load_dir + file_name)
        # First check if this is an old pickle version, if so transform it into a df
        if type(df) != pd.DataFrame:
            loaded_par = df
            df = pd.DataFrame(columns=['key', 'value'])
            for key, v in loaded_par.__dict__.items():
                try:
                    for key2, vv in v.__dict__.items():
                        temp = pd.DataFrame(data=[str(key) + '_' + str(key2), vv], index=['key', 'value']).T
                        df = df.append(temp)
                except:
                    temp = pd.DataFrame(data=[key, v], index=['key', 'value']).T
                    df = df.append(temp)

        no_old_version_bug = True

        for key, v in self.__dict__.items():
            try:
                for key2, vv in v.__dict__.items():
                    t = df.loc[df['key'] == str(key) + '_' + str(key2), 'value']
                    if t.shape[0] == 1:
                        tt = t.values[0]
                        self.__dict__[key].__dict__[key2] = tt
                    else:
                        if no_old_version_bug:
                            no_old_version_bug = False
                            print('#### Loaded basic_parameters object is depreceated, default version will be used')
                        print('Parameter', str(key) + '.' + str(key2), 'not found, using default: ',
                              self.__dict__[key].__dict__[key2])

            except:
                t = df.loc[df['key'] == str(key), 'value']
                if t.shape[0] == 1:
                    tt = t.values[0]
                    self.__dict__[key] = tt
                else:
                    if no_old_version_bug:
                        no_old_version_bug = False
                        print('#### Loaded basic_parameters object is depreceated, default version will be used')
                    print('Parameter', str(key), 'not found, using default: ', self.__dict__[key])



    def dict_to_string_for_dir(self, d:dict, old_style =False):
        if (self.use_hash) & (old_style==False):
            valid_params = {k: v for k, v in d.items() if v is not None}
            # Convert the dictionary to a string representation
            param_string = str(valid_params)

            # Create a hash of the string
            hash_object = hashlib.sha256(param_string.encode())
            s = hash_object.hexdigest()
        else:
            # the old version for backward compatibility
            s = ''
            for k in d.keys():
                if d[k] is not None:
                    # we only add to the string name if a parameters is not none. That allows us to keep compatible stuff with old models by adding new parameters with None
                    # v= d[k] if type(d[k]) not in [type(np.array([])),type([])] else len(d[k])
                    if type(d[k]) in [type(np.array([])), type([])]:
                        v = len(d[k])
                        if v == 1:
                            v = d[k][0]
                    else:
                        v = d[k]
                    s += f'{k}{v}'
        return s

if __name__ == '__main__':
    self = Params()

