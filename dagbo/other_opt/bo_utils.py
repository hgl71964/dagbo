import botorch
import gpytorch
from torch import Tensor
from botorch.models import SingleTaskGP

from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax import SearchSpace, Experiment, OptimizationConfig, Runner, Objective

from dagbo.other_opt.model_factory import make_gps, fit_gpr
from dagbo.other_opt.acq_func_factory import opt_acq_func
from dagbo.utils.ax_experiment_utlis import (get_tensor, get_bounds,
                                             print_experiment_result, save_exp,
                                             load_exp)


def get_fitted_model(exp: Experiment, params: list[str]) -> SingleTaskGP:
    """instantiate and fit a gp"""

    x, y = get_tensor(exp, params)
    gpr = make_gps(x=x, y=y, gp_name="MA")
    fit_gpr(gpr)
    return gpr


def inner_loop(exp: Experiment, model: SingleTaskGP, params: list[str],
               acq_name: str, acq_func_config: dict) -> Tensor:
    """acquisition function optimisation"""
    bounds = get_bounds(exp, params)
    return opt_acq_func(model, acq_name, bounds, acq_func_config)


def candidates_to_generator_run(exp: Experiment, candidate: Tensor,
                                params: list[str]) -> GeneratorRun:
    """
    Args:
        candidate: [q, dim]
    """
    n = exp.num_trials
    q = candidate.shape[0]
    arms = []
    for i in range(q):
        p = {}
        for j, name in enumerate(params):
            p[name] = float(candidate[
                i,
                j])  # need to convert back to python type, XXX not support int
        arms.append(Arm(parameters=p, name=f"{n}_{i}"))
    return GeneratorRun(arms=arms)
