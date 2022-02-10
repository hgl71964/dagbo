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


def get_fitted_model(exp: Experiment, params: list[str],
                     dtype) -> SingleTaskGP:
    """ `freshly` instantiate and fit a gp"""
    x, y = get_tensor(exp, params, dtype)
    gpr = make_gps(x=x, y=y, gp_name="MA")
    fit_gpr(gpr)
    return gpr


def inner_loop(exp: Experiment, model: Union[Dag,
                                             SingleTaskGP], params: list[str],
               acq_name: str, acq_func_config: dict, dtype) -> Tensor:
    """acquisition function optimisation"""
    bounds = get_bounds(exp, params, dtype)
    return opt_acq_func(model, acq_name, bounds, acq_func_config)
