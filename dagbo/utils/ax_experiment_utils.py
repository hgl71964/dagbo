import os
import pickle
from os.path import join, abspath, exists
import torch
import pandas as pd
from torch import Tensor
from copy import deepcopy
from typing import Union

import ax
from ax import ParameterType
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun
from ax import SearchSpace, Experiment, OptimizationConfig, Runner, Objective
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment
"""
the order of params in candidates_to_generator_run,
                        get_tensor_to_dict,
                        get_tensor,
                        get_bounds,
    MUST be equal
"""


def candidates_to_generator_run(exp: Experiment, candidate: Tensor,
                                params: list[str]) -> GeneratorRun:
    """
    user-defined data type -> arms -> generator_run -> trial.run()
    Args:
        candidate: [q, dim]
    """
    q = candidate.shape[0]
    arms = []
    for i in range(q):
        p = {}
        for j, name in enumerate(params):
            # need to convert back to python type, XXX not support int
            p[name] = float(candidate[i, j])

        # TODO make sure arm has unique signature?
        arms.append(Arm(parameters=p))
        #arms.append(Arm(parameters=p, name=f"bo_{n}_{i}"))
    return GeneratorRun(arms=arms)


def get_dict_tensor(
    exp: Experiment,
    params: list[str],
    dtype,
) -> dict[str, Tensor]:
    """retrieve data from experiment to tensor
    single objective ONLY

    Args:
        exp (Experiment): Ax.Experiment
        params (list): param name str list

    Returns:
        key: param name - val: Tensor
    """

    exp_df = exp.fetch_data().df
    train_inputs_dict = {}

    # follow trials order
    num_trials = exp_df.shape[0]
    arm_name_list = list(exp_df["arm_name"])  # [num_trials, ]

    # retrieve data from experiment
    for arm_name in arm_name_list:
        arm_ = exp.arms_by_name[arm_name]
        arm_param = arm_.parameters
        for p in params:
            val = deepcopy(arm_param[p])

            if p in train_inputs_dict:
                train_inputs_dict[p].append(arm_param[p])
            else:
                train_inputs_dict[p] = [arm_param[p]]

    # convert to tensor
    for key in train_inputs_dict:
        train_inputs_dict[key] = torch.tensor(train_inputs_dict[key],
                                              dtype=dtype)
    return train_inputs_dict


def get_tensor(exp: Experiment, params: list[str],
               dtype) -> tuple[Tensor, Tensor]:
    """retrieve data from experiment to tensor
    single objective ONLY

    Args:
        exp (Experiment): Ax.Experiment
        params (list): param name str list

    Returns:
        x: [num_trials, dim_arm]
        y: [num_trials, 1]
    """
    _check_name_consistency(exp.parameters)
    exp_df = exp.fetch_data().df

    # follow trials order
    num_trials = exp_df.shape[0]
    rewards = torch.tensor(exp_df["mean"],
                           dtype=dtype).reshape(-1, 1)  # [num_trials, 1]
    arm_name_list = list(exp_df["arm_name"])  # [num_trials, ]

    data = []
    for arm_name in arm_name_list:
        arm_ = exp.arms_by_name[arm_name]
        arm_param = arm_.parameters
        for p in params:
            val = deepcopy(arm_param[p])
            data.append(val)

    # [num_trials, dim_arm]
    t = torch.tensor(data, dtype=dtype).reshape(num_trials, -1)
    return t, rewards


def get_bounds(exp: Experiment, params: list[str], dtype) -> Tensor:
    """get bounds for each parameters"""
    bounds = []
    for p in params:
        ax_param = exp.parameters[p]
        bounds.append(ax_param.lower)
        bounds.append(ax_param.upper)

    return torch.tensor(bounds, dtype=dtype).reshape(-1, 2).T


def print_experiment_result(exp: Experiment) -> None:
    """print experiment metric + arms"""
    df = exp.fetch_data().df.set_index("arm_name")
    arms_df = pd.DataFrame.from_dict(
        {k: v.parameters
         for k, v in exp.arms_by_name.items()}, orient="index")
    return df.join(arms_df)


def save_exp(exp: Experiment, name: str) -> None:
    directory = os.path.dirname(__file__)
    data_dir = join(directory, "../../benchmarks/data")
    file_name = name + ".json"
    full_path = join(data_dir, file_name)

    if exists(full_path):
        print(f"Experiment {file_name} exists!")
        return None

    save_experiment(exp, full_path)
    print(f"save as {name}.json")
    return None


def save_dict(train_targets_dict: Union[dict, list[dict]], name: str) -> None:
    directory = os.path.dirname(__file__)
    data_dir = join(directory, "../../benchmarks/data")
    file_name = name + ".pkl"
    full_path = join(data_dir, file_name)

    if exists(full_path):
        print(f"dict {file_name} exists!")
        return None

    with open(full_path, "wb") as f:
        pickle.dump(train_targets_dict, f)
    return None


def load_exp(name: str) -> Experiment:
    directory = os.path.dirname(__file__)
    data_dir = join(directory, "../../benchmarks/data")
    file_name = name + ".json"
    print(f"load from {name}.json")
    return load_experiment(join(data_dir, file_name))


def load_dict(name: str) -> Union[dict, list[dict]]:
    directory = os.path.dirname(__file__)
    data_dir = join(directory, "../../benchmarks/data")
    file_name = name + ".pkl"
    full_path = join(data_dir, file_name)
    with open(full_path, "rb") as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def _check_name_consistency(all_params):
    for k, v in all_params.items():
        if k != v.name:
            raise NameError(
                f"parameter {k} and name {v.name} is not consistent")
