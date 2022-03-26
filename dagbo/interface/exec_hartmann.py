import os
from typing import Union
import numpy as np

# NOTE: possibly the most important mapping, scale param range [0, 1] back to their original values
SCALE_MAPPING = {
        "p": 1,
        }


def call_hartmann(
        params: dict[str, float], train_inputs_dict: dict[str, np.ndarray],
        train_targets_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    6-dim hartmann benchmark func
        1. minimization
        2. global minimum:  -3.32237
        3. x* = [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)]
    """

    # NOTE: scale back because params are always defined within [0, 1]
    scale = SCALE_MAPPING["p"]
    n = len(params.keys())
    assert n == 6, "hartmann should be 6-dim"
    alphas = [1.0, 1.2, 3.0, 3.2]
    A = [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
            ]
    P = [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
            ]
    exp = [0 for i in range(4)]
    for i in range(4):
        tmp = []
        for j in range(6):
            xj = params[f"x{j}"]
            inner = A[i][j] * (xj - 0.0001 * P[i][j]) ** 2
            tmp.append(inner)
        exp[i] = alphas[i] * np.exp(-sum(tmp))

    obj = {}
    for c, item in enumerate(exp):
        obj[f"exp{c}"] = np.array([item]).reshape(-1)
    obj["final"] = -np.array([sum(exp)]).reshape(-1)

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
    params = {}
    x_star = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
    for i in range(6):
        params[f"x{i}"] = x_star[i]
    print(call_hartmann(params, {}, {}))
