import os
import sys
from typing import List, Dict

import torch
from torch import Tensor

import ax
from ax import SearchSpace, Experiment, OptimizationConfig, Runner, Objective
from ax.core.arm import Arm
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
from ordinary_bo.ax_experiment_utlis import get_tensor, get_bounds, print_experiment_result




if __name__ == "__main__":
    # TODO load experiment
    print()