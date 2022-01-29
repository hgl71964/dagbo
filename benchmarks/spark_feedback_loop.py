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
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.runners.synthetic import SyntheticRunner


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

class SparkMetric(Metric):
    def fetch_trial_data(self, trial, **kwargs):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": 1, # TODO
                "sem": 0,  # 0 for noiseless experiment
                "trial_index": trial.index,
            })
        return ax.core.data.Data(df=pd.DataFrame.from_records(records))

# TODO
# NOTE: this must
param_names = ["x1", "x2", "x3"]

search_space = SearchSpace([RangeParameter("x1", ParameterType.FLOAT, lower=-1, upper=1),
    RangeParameter("x2", ParameterType.FLOAT, lower=-1, upper=1),
    RangeParameter("x3", ParameterType.FLOAT, lower=-1, upper=1),
    ])

# opt config
optimization_config = OptimizationConfig(
    Objective(metric=CustomMetric(name="custom_obj"), minimize=False))

# experiment
exp = Experiment(name="test_exp",
                 search_space=self.search_space,
                 optimization_config=self.optimization_config,
                 runner=SyntheticRunner())

# BOOTSTRAP EVALUATIONS
num_bootstrap = 2
sobol = Models.SOBOL(self.exp.search_space)
generated_run = sobol.gen(num_bootstrap)
trial = self.exp.new_batch_trial(generator_run=generated_run)
trial.run()
trial.mark_completed()

epoch = 3


def main(_):
    # build experiment
    exp
    param_names

    # get dag's spec
    param_space, metric_space, obj_space, edges = parse_model(FLAGS.performance_model_path)

    # bo loop
    for t in range(FLAGS.epochs):
        # input params can be read from ax experiment
        train_inputs_dict = input_dict_from_ax_experiment(exp, param_names)

        # for make-up metric, must build by hand...
        train_targets_dict = {
                }


        # fit model from dataset
        dag = build_perf_model_from_spec(train_inputs_dict,
                                   train_targets_dict,
                                   acq_func_config["num_samples"],
                                   param_space,
                                   metric_space,
                                   obj_space,
                                   edges)
        fit_dag(dag)

        # get candidates (inner loop)
        candidates = inner_loop(exp,
                                dag,
                                param_names,
                                acq_name="qUCB",
                                acq_func_config=acq_func_config)
        gen_run = candidates_to_generator_run(exp, candidates, param_names)

        # run
        if self.acq_func_config["q"] == 1:
            trial = exp.new_trial(generator_run=gen_run)
        else:
            trial = exp.new_batch_trial(generator_run=gen_run)
        trial.run()
        trial.mark_completed()

    print("done")
    print(exp.fetch_data().df)



if __name__ == "__main__":
    app.run(main)
