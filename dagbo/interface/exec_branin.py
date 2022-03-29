import os
from typing import Union
import numpy as np
import math

# NOTE: possibly the most important mapping, scale param range [0, 1] back to their original values
SCALE_MAPPING = {
    "p": 15,
}


def call_branin(
        params: dict[str, float], train_inputs_dict: dict[str, np.ndarray],
        train_targets_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    2-dim branin benchmark func
        1. minimization
        2. global minimum:  0.397887
        3. x* = [(-pi, 12.275), (pi, 2.275), (9.42, 2.475)]
        4. x0 in [-5, 10], x1 in [0, 15]
    """

    # NOTE: scale back because params are always defined within [0, 1]
    scale = SCALE_MAPPING["p"]
    n = len(params.keys())
    assert n == 2, "branin should be 2-dim"

    x0 = params["x0"] * 15 - 5
    x1 = params["x1"] * 15
    #x0 = params["x0"]
    #x1 = params["x1"]

    t1 = x1 - 5.1 / (4 * math.pi**2) * (x0)**2 + 5 / math.pi * x0 - 6
    square = t1**2
    t2 = 10 * (1 - 1 / (8 * math.pi)) * np.cos(x0)

    obj = {}
    obj[f"t1"] = np.array([t1]).reshape(-1)
    obj[f"t2"] = np.array([t2]).reshape(-1)
    obj[f"square"] = np.array([square]).reshape(-1)
    obj["final"] = square + t2 + 10

    # populdate input dict
    for k, v in params.items():
        if k in train_inputs_dict:
            train_inputs_dict[k] = np.append(train_inputs_dict[k], v)
        else:
            train_inputs_dict[k] = np.array([v])

    #  populate output dict
    for k, v in obj.items():
        # add to target dict
        if k in train_targets_dict:
            train_targets_dict[k] = np.append(train_targets_dict[k], v)
        else:
            train_targets_dict[k] = np.array([v])

    return obj


if __name__ == "__main__":
    # NOTE: this only works if we disable the scale
    params = {}
    x_star = (math.pi, 2.275)
    for i in range(2):
        params[f"x{i}"] = x_star[i]
    print(call_branin(params, {}, {}))
    x_star = (-math.pi, 12.275)
    for i in range(2):
        params[f"x{i}"] = x_star[i]
    print(call_branin(params, {}, {}))
    x_star = (9.42478, 2.475)
    for i in range(2):
        params[f"x{i}"] = x_star[i]
    print(call_branin(params, {}, {}))
