import os
import sys
from typing import list, dict

import torch
from torch import Tensor

import ax
from ax import SearchSpace, Experiment, OptimizationConfig, Runner, Objective
from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.modelbridge.registry import Models
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner

import botorch
from botorch.models import SingleTaskGP

# hacky way to include the src code dir
testdir = os.path.dirname(__file__)
srcdir = "../src"
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

from ordinary_bo.model_factory import make_gps, fit_gpr
from ordinary_bo.acq_func_factory import opt_acq_func
from ordinary_bo.ax_experiment_utlis import (get_tensor, get_bounds,
                                             print_experiment_result, save_exp,
                                             load_exp)
"""
Create an experiment and initial SOBOL points,
    and save this experiment 

so that all algorithms have the same initial conditions by loading this experiment
"""


class MyRunner(Runner):
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        return trial_metadata


def get_fitted_model(exp: Experiment, params: list) -> SingleTaskGP:
    """instantiate and fit a gp"""

    x, y = get_tensor(exp, params)
    gpr = make_gps(x=x, y=y, gp_name="MA")
    fit_gpr(gpr)
    return gpr


def inner_loop(exp: Experiment, model: SingleTaskGP, params: list,
               acq_name: str, acq_func_config: dict) -> Tensor:
    """acquisition function optimisation"""
    bounds = get_bounds(exp, params)
    return opt_acq_func(model, acq_name, bounds, acq_func_config)


def candidates_to_generator_run(exp: Experiment, candidate: Tensor,
                                params: list) -> GeneratorRun:
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


if __name__ == "__main__":
    NUM_SOBOL_TRIALS = 5
    NUM_BOTORCH_TRIALS = 7
    acq_func_config = {
        "q": 2,
        "num_restarts": 48,
        "raw_samples": 128,
        "num_samples": 2048,
        "y_max": torch.tensor([1.]),  # for EI
        "beta": 1,
    }
    param_names = [f"x{i}" for i in range(6)]  # must respect order
    hartmann_search_space = SearchSpace(parameters=[
        ax.RangeParameter(name=f"x{i}",
                          parameter_type=ax.ParameterType.FLOAT,
                          lower=0.0,
                          upper=1.0) for i in range(6)
    ])
    optimization_config = OptimizationConfig(objective=Objective(
        metric=Hartmann6Metric(name="hartmann6", param_names=param_names),
        minimize=True,
    ), )
    exp = Experiment(
        name="test_hartmann",
        search_space=hartmann_search_space,
        optimization_config=optimization_config,
        runner=MyRunner(),
    )

    # register for saving
    register_metric(Hartmann6Metric)
    register_runner(MyRunner)

    print(f"Running Sobol initialization trials...")
    sobol = Models.SOBOL(search_space=exp.search_space)
    for i in range(NUM_SOBOL_TRIALS):
        generator_run = sobol.gen(n=1)
        trial = exp.new_trial(generator_run=generator_run)
        trial.run()
        trial.mark_completed()

    save_exp(exp, "basic")
    #for i in range(NUM_BOTORCH_TRIALS):
    #    print(
    #        f"Running optimization trial {i + NUM_SOBOL_TRIALS + 1}/{NUM_SOBOL_TRIALS + NUM_BOTORCH_TRIALS}..."
    #    )
    #    """custom impl of BO component"""
    #    model = get_fitted_model(exp, param_names)
    #    candidates = inner_loop(exp,
    #                            model,
    #                            param_names,
    #                            acq_name="qUCB",
    #                            acq_func_config=acq_func_config)
    #    gen_run = candidates_to_generator_run(exp, candidates, param_names)
    #    """ax APIs"""
    #    if acq_func_config["q"] == 1:
    #        trial = exp.new_trial(generator_run=gen_run)
    #    else:
    #        trial = exp.new_batch_trial(generator_run=gen_run)
    #    trial.run()
    #    trial.mark_completed()
    #"""analysis"""
    #print()
    #print("Done!")
    #print(exp.fetch_data().df)
    #print(print_experiment_result(exp))
