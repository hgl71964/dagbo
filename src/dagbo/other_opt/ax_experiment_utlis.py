import os
from os.path import join, abspath, exists
import torch
import pandas as pd
from torch import Tensor
from typing import dict, list, tuple
from copy import deepcopy

import ax
from ax import SearchSpace, Experiment, OptimizationConfig, Runner, Objective
from ax import ParameterType
from ax.core.generator_run import GeneratorRun
from ax.core.data import Data
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment


def get_tensor(exp: Experiment, params: list) -> tuple[Tensor, Tensor]:
    """convert data from experiment to tensor
    single objective ONLY

    Args:
        exp (Experiment): Ax.Experiment
        params (list): MUST be orderred

    Returns:
        x: [num_arms, dim_arm]
        y: [reward, 1]
    """
    _check_name_consistency(exp.parameters)

    data = []
    num_arms = len(exp.arms_by_name)
    for _, arm in exp.arms_by_name.items():
        for p in params:
            val = deepcopy(arm.parameters[p])
            data.append(val)
    #print(data)

    # [num_arms, dim_arm]
    return torch.tensor(data, dtype=torch.float32).reshape(num_arms, -1), \
                torch.tensor(exp.fetch_data().df["mean"], dtype=torch.float32).reshape(-1, 1)


def get_bounds(exp: Experiment, params: list) -> Tensor:
    """get bounds for each parameters"""
    bounds = []
    for p in params:
        ax_param = exp.parameters[p]
        bounds.append(ax_param.lower)
        bounds.append(ax_param.upper)

    # XXX bounds should be set as float?
    return torch.tensor(bounds, dtype=torch.float32).reshape(-1, 2).T


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


def load_exp(name: str) -> Experiment:
    directory = os.path.dirname(__file__)
    data_dir = join(directory, "../../benchmarks/data")
    file_name = name + ".json"
    print(f"load from {name}.json")
    return load_experiment(join(data_dir, file_name))


def _check_name_consistency(all_params):
    for k, v in all_params.items():
        if k != v.name:
            raise NameError(
                f"parameter {k} and name {v.name} is not consistent")
