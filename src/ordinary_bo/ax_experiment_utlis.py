import ax
import torch
from ax import SearchSpace, Experiment, OptimizationConfig, Runner, Objective
from ax.core.generator_run import GeneratorRun
from ax.core.data import Data
from torch import Tensor
from typing import Dict, List, Tuple
from copy import deepcopy


def exp2tensor(exp: Experiment, params: List) -> Tuple[Tensor, Tensor]:
    """convert data from experiment to tensor
    single objective ONLY

    Args:
        exp (Experiment): Ax.Experiment
        params (List): MUST be orderred

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


def exp2bounds(exp: Experiment, params: List) -> Tuple[Tensor, Tensor]:
    return None


def _check_name_consistency(all_params):
    for k, v in all_params.items():
        if k != v.name:
            raise NameError(
                f"parameter {k} and name {v.name} is not consistent")
