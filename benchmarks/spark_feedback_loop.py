import os
import sys
from absl import app
from absl import flags

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

from dagbo.other_opt.model_factory import make_gps, fit_gpr
from dagbo.other_opt.acq_func_factory import opt_acq_func
from dagbo.other_opt.ax_experiment_utlis import (get_tensor, get_bounds,
                                             print_experiment_result, save_exp,
                                             load_exp)
"""
Create an experiment and initial SOBOL points,
    and save this experiment

so that all algorithms have the same initial conditions by loading this experiment
"""
FLAGS = flags.FLAGS
flags.DEFINE_string('name', 'Jane Random', 'Your name.')
flags.DEFINE_integer('age', None, 'Your age in years.', lower_bound=0)
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.DEFINE_enum('job', 'running', ['running', 'stopped'], 'Job status.')

# flags cannot define dict
acq_func_config = {
    "q": 2,
    "num_restarts": 48,
    "raw_samples": 128,
    "num_samples": 2048,
    "y_max": torch.tensor([1.]),  # for EI
    "beta": 1,  # for UCB
}

def main(_):
    build_spark_dag()
    # TODO

    pass

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
    app.run(main)
