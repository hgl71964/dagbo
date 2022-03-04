from typing import Union

from ax import Experiment
import torch
import botorch
import gpytorch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.sampling.samplers import MCSampler
from botorch.sampling.samplers import SobolQMCNormalSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dagbo.dag import Dag
from dagbo.utils.perf_model_utils import get_dag_topological_order


def inner_loop(exp: Experiment,
               model: Union[Dag, SingleTaskGP],
               param_space: dict,
               obj_space: dict,
               acq_name: str,
               acq_func_config: dict,
               dtype=torch.float64) -> Tensor:
    """acquisition function optimisation"""

    bounds = get_bounds(exp, param_space, dtype)
    return opt_acq_func(model, acq_name, bounds, acq_func_config)


def get_bounds(exp: Experiment,
               param_space: dict,
               dtype=torch.float64) -> Tensor:
    """get bounds for each parameters"""
    params = sorted(list(param_space.keys()))

    bounds = []
    for p in params:
        ax_param = exp.parameters[p]
        bounds.append(ax_param.lower)
        bounds.append(ax_param.upper)

    return torch.tensor(bounds, dtype=dtype).reshape(-1, 2).T


def opt_acq_func(model: Union[SingleTaskGP, Dag], acq_name: str,
                 bounds: Tensor, acq_func_config: dict) -> Tensor:

    sampler = make_sampler(acq_func_config)
    acq_func = make_acq_func(model, acq_name, sampler, acq_func_config)
    """
    by default, acq_func is set to be maximised
        so candidates will try to maximise the model
    """

    candidates, _ = botorch.optim.optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=acq_func_config["q"],
        num_restarts=acq_func_config["num_restarts"],
        raw_samples=acq_func_config["raw_samples"],
        sequential=False,
    )
    query = candidates.detach()
    return query


def make_acq_func(model: SingleTaskGP, acq_name: str, sampler: MCSampler,
                  acq_func_config: dict) -> AcquisitionFunction:
    if acq_name == "qKG":
        acq = botorch.acquisition.qKnowledgeGradient(
            model=model,
            num_fantasies=acq_func_config.get("num_fantasies", 128),
            sampler=sampler,
            objective=None,
        )
    elif acq_name == "qEI":
        acq = botorch.acquisition.monte_carlo.qExpectedImprovement(
            model=model,
            best_f=acq_func_config["y_max"],
            sampler=sampler,
            objective=None,
        )
    elif acq_name == "qUCB":
        acq = botorch.acquisition.monte_carlo.qUpperConfidenceBound(
            model=model,
            beta=acq_func_config["beta"],
            sampler=sampler,
            objective=None,
        )
    elif acq_name == "EI":
        acq = botorch.acquisition.analytic.ExpectedImprovement(
            model=model,
            best_f=acq_func_config["y_max"],
            objective=None,
        )
    elif acq_name == "UCB":
        acq = botorch.acquisition.analytic.UpperConfidenceBound(
            model=model,
            beta=acq_func_config["beta"],
            objective=None,
        )
    else:
        raise NameError("acquisition function name not recognise")

    return acq


def make_sampler(acq_func_config: dict) -> MCSampler:
    """SBO should always use quasi-MC sampler"""
    return SobolQMCNormalSampler(num_samples=acq_func_config["num_samples"],
                                 seed=acq_func_config.get("seed", 1234))
