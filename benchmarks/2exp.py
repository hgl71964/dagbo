import os
import sys
from typing import List, Dict
from torch import Tensor

import ax
from ax import SearchSpace, Experiment, OptimizationConfig, Runner, Objective
from ax.core.generator_run import GeneratorRun
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.modelbridge.registry import Models

import botorch
from botorch.models import SingleTaskGP

# hacky way to include the src code dir
testdir = os.path.dirname(__file__)
srcdir = "../src"
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

from ordinary_bo.model_factory import make_gps, fit_gpr
from ordinary_bo.acq_func_factory import opt_acq_func
from ordinary_bo.ax_experiment_utlis import exp2tensor, exp2bounds


class MyRunner(Runner):
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        return trial_metadata


def get_fitted_model(exp: Experiment, params: List) -> SingleTaskGP:
    """instantiate and fit a gp"""

    x, y = exp2tensor(exp, params)
    gpr = make_gps(x=x, y=y, name="MA")
    fit_gpr(gpr)
    return gpr


def inner_loop(exp: Experiment, model: SingleTaskGP, params: List,
               acq_name: str, acq_func_config: Dict) -> Tensor:
    """acquisition function optimisation"""
    exp2bounds()
    return


def candidates_to_generator_run(candidate: Tensor) -> GeneratorRun:
    return


if __name__ == "__main__":
    NUM_SOBOL_TRIALS = 5
    NUM_BOTORCH_TRIALS = 15
    acq_func_config = {
        "q": 2,
        # TODO
    }
    hartmann_search_space = SearchSpace(parameters=[
        ax.RangeParameter(name=f"x{i}",
                          parameter_type=ax.ParameterType.FLOAT,
                          lower=0.0,
                          upper=1.0) for i in range(6)
    ])
    param_names = [f"x{i}" for i in range(6)]  # must respect order
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

    print(f"Running Sobol initialization trials...")
    sobol = Models.SOBOL(search_space=exp.search_space)
    for i in range(NUM_SOBOL_TRIALS):
        generator_run = sobol.gen(n=1)
        trial = exp.new_trial(generator_run=generator_run)
        trial.run()
        trial.mark_completed()

    for i in range(NUM_BOTORCH_TRIALS):
        print(
            f"Running optimization trial {i + NUM_SOBOL_TRIALS + 1}/{NUM_SOBOL_TRIALS + NUM_BOTORCH_TRIALS}..."
        )
        #gpei = Models.BOTORCH(experiment=exp, data=exp.fetch_data())
        #generator_run = gpei.gen(n=1)
        model = get_fitted_model(exp, param_names)
        # TODO
        inner_loop(model, "qEI")
        generator_run = candidates_to_generator_run()
        trial = exp.new_trial(generator_run=generator_run)
        trial.run()
        trial.mark_completed()

    print("Done!")
