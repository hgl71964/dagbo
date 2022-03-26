import os
from typing import Union
import numpy as np

# NOTE: possibly the most important mapping, scale param range [0, 1] back to their original values
SCALE_MAPPING = {
    "p": 3,
}


def call_rosenbrock(
        params: dict[str, float], train_inputs_dict: dict[str, np.ndarray],
        train_targets_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    n-dim rosenbrock benchmark func, n is automatically inferred from params
        1. minimization
		2. global minimum: 0
        3. x* = [(1, ...)]
    """

    # NOTE: scale back because params are always defined within [0, 1]
    scale = SCALE_MAPPING["p"]
    n = len(params.keys())
    i_s = []
    f_s = []
    for i in range(n - 1):
        x_cur = params[f"x{i}"] * scale
        x_next = params[f"x{i+1}"] * scale

        tmp = 100 * (x_next - x_cur**2)**2
        tmp_2 = tmp + (1 - x_cur)**2

        i_s.append(tmp)
        f_s.append(tmp_2)

    # populate intermediate metric
    obj = {}
    for c, (i, j) in enumerate(zip(i_s, f_s)):
        obj[f"i{c}"] = np.array([i]).reshape(-1)
        obj[f"f{c}"] = np.array([j]).reshape(-1)
    # final obj
    obj["final"] = np.array([sum(f_s)]).reshape(-1)

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
