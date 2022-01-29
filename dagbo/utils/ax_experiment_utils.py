import os
from os.path import join, abspath, exists
import torch
import pandas as pd
from torch import Tensor
from copy import deepcopy

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
    Args:
        candidate: [q, dim]
    """
    n = exp.num_trials + 1
    q = candidate.shape[0]
    arms = []

    # FIXME a bug in trial naming
    print(exp.trials)
    print(n)

    for i in range(q):
        p = {}
        for j, name in enumerate(params):
            # need to convert back to python type, XXX not support int
            p[name] = float(candidate[i, j])
        arms.append(Arm(parameters=p, name=f"{n}_{i}"))
    return GeneratorRun(arms=arms)


def get_tensor(exp: Experiment, params: list[str]) -> tuple[Tensor, Tensor]:
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
                           dtype=torch.float32).reshape(-1,
                                                        1)  # [num_trials, 1]
    arm_name_list = list(exp_df["arm_name"])  # [num_trials, ]

    data = []
    for arm_name in arm_name_list:
        arm_ = exp.arms_by_name[arm_name]
        arm_param = arm_.parameters
        for p in params:
            val = deepcopy(arm_param[p])
            data.append(val)

    # [num_trials, dim_arm]
    t = torch.tensor(data, dtype=torch.float32).reshape(num_trials, -1)
    return t, rewards


def get_bounds(exp: Experiment, params: list[str]) -> Tensor:
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
