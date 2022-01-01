import os
import sys

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

from basic_exp import get_fitted_model, inner_loop, candidates_to_generator_run


class MyRunner(Runner):
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        return trial_metadata


if __name__ == "__main__":
    # load
    param_names = [f"x{i}" for i in range(6)]  # must respect order
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
    exp_name = "basic"
    register_metric(Hartmann6Metric)
    register_runner(MyRunner)
    exp = load_exp(exp_name)
    """benchmarking"""
    for i in range(NUM_BOTORCH_TRIALS):
        print(
            f"Running optimization trial {i + NUM_SOBOL_TRIALS + 1}/{NUM_SOBOL_TRIALS + NUM_BOTORCH_TRIALS}..."
        )
        """custom impl of BO component"""
        model = get_fitted_model(exp, param_names)
        candidates = inner_loop(exp,
                                model,
                                param_names,
                                acq_name="qUCB",
                                acq_func_config=acq_func_config)
        gen_run = candidates_to_generator_run(exp, candidates, param_names)
        """ax APIs"""
        if acq_func_config["q"] == 1:
            trial = exp.new_trial(generator_run=gen_run)
        else:
            trial = exp.new_batch_trial(generator_run=gen_run)
        trial.run()
        trial.mark_completed()
    """analysis"""
    print()
    print("Done!")
    print(exp.fetch_data().df)
    print(print_experiment_result(exp))
