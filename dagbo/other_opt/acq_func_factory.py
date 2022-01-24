import botorch
import gpytorch
from torch import Tensor
from typing import dict
from botorch.models import SingleTaskGP
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.sampling.samplers import MCSampler
from botorch.sampling.samplers import SobolQMCNormalSampler


def opt_acq_func(model: SingleTaskGP, acq_name: str, bounds: Tensor,
                 acq_func_config: dict) -> Tensor:

    sampler = make_sampler(acq_func_config)
    acq_func = make_acq_func(model, acq_name, sampler, acq_func_config)

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
            num_fantasies=128,
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
                                 seed=1234)
