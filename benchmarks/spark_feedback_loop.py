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

from dagbo.interface.parse_performance_model import parse_model
from dagbo.utils.perf_model_utlis import build_perf_model_from_spec
from dagbo.fit_dag import fit_dag

"""
run the whole spark feedback loop
"""

FLAGS = flags.FLAGS
flags.DEFINE_string("performance_model_path", "dagbo/interface/spark_performance_model.txt", "graphviz source path")
flags.DEFINE_integer("epochs", 5, "bo loop epoch", lower_bound=0)
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

def fit_model():
    return

def get_candidate():
    return

def main(_):

    # build performance dag
    param_space, metric_space, obj_space, edges = parse_model(FLAGS.performance_model_path)

    # make fake input tensor
    train_inputs_dict = {i: torch.rand(acq_func_config["q"]) for i in list(param_space.keys())}
    train_targets_dict = {i: torch.rand(acq_func_config["q"]) for i in list(metric_space.keys()) + list(obj_space.keys())}

    dag = build_perf_model_from_spec(train_inputs_dict,
                               train_targets_dict,
                               acq_func_config["num_samples"],
                               param_space,
                               metric_space,
                               obj_space,
                               edges)

    # bo loop
    for t in range(FLAGS.epochs):
        # fit model from dataset
        fit_dag(dag)

        # get candidates (inner loop)

        # query system

        # update dataset



if __name__ == "__main__":
    app.run(main)
