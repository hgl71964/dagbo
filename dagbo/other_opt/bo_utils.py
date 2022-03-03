from typing import Union
import botorch
import gpytorch
from torch import Tensor
from botorch.models import SingleTaskGP
from ax import Experiment

from dagbo.dag import Dag
from dagbo.other_opt.model_factory import make_gps, fit_gpr
from dagbo.other_opt.acq_func_factory import opt_acq_func
from dagbo.utils.ax_experiment_utils import get_tensor, get_bounds

# deprecated
#def get_fitted_model(exp: Experiment, params: list[str],
#                     dtype) -> SingleTaskGP:
#    """ `freshly` instantiate and fit a gp"""
#        x: [num_trials, dim_arm]
#        y: [num_trials, 1]
#    x, y = get_tensor(exp, params, dtype)
#    gpr = make_gps(x=x, y=y, gp_name="MA")
#    fit_gpr(gpr)
#    return gpr


def build_gp_from_spec(train_inputs_dict: dict[str, Tensor],
                       train_targets_dict: dict[str, Tensor],
                       param_space: dict[str, str], metric_space: dict[str,
                                                                       str],
                       obj_space: dict[str, str], edges: dict[str, list[str]],
                       normalisation: Union[bool, dict]):

    # TODO make input to gp
    if isinstance(normalisation, dict):
        raise TypeError(f"unsupported norm type")
    elif isinstance(normalisation, bool):
        if normalisation:
            pass
        else:
            pass
    else:
        raise TypeError(f"unrecognized normalization type")

    gpr = make_gps(x=x, y=y, gp_name="MA")
    return gpr


def inner_loop(exp: Experiment, model: Union[Dag,
                                             SingleTaskGP], params: list[str],
               acq_name: str, acq_func_config: dict, dtype) -> Tensor:
    """acquisition function optimisation"""
    bounds = get_bounds(exp, params, dtype)
    return opt_acq_func(model, acq_name, bounds, acq_func_config)
