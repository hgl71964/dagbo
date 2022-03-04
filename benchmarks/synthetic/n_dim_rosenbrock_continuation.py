import time
from absl import app
from absl import flags
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from botorch.models import SingleTaskGP

import ax
from ax.modelbridge.registry import Models
from ax import SearchSpace, Experiment, OptimizationConfig, Objective, Metric
from ax.storage.metric_registry import register_metric
from ax.runners.synthetic import SyntheticRunner

from dagbo.models.model_builder import build_model
from dagbo.models.acq_func import inner_loop
from dagbo.utils.ax_experiment_utils import (candidates_to_generator_run,
                                             load_exp, load_dict,
                                             print_experiment_result,
                                             save_dict, save_exp)
from dagbo.interface.exec_n_dim_rosenbrock import call_rosenbrock
from dagbo.interface.parse_performance_model import parse_model
"""
load an experiment with initial sobol points & run opt loop
"""

FLAGS = flags.FLAGS
flags.DEFINE_enum("tuner", "dagbo-ssa", ["dagbo-direct", "dagbo-ssa", "bo"],
                  "tuner to use")
flags.DEFINE_string("exp_name", "SOBOL-spark-wordcount", "Experiment name")
flags.DEFINE_string("load_name", "SOBOL-spark-wordcount",
                    "load from experiment name")
flags.DEFINE_string("acq_name", "qEI", "acquisition function name")
flags.DEFINE_string("performance_model_path", "must provide",
                    "graphviz source path")

flags.DEFINE_integer("n_dim", 10, "n-dim rosenbrock func")
flags.DEFINE_integer("epochs", 20, "bo loop epoch", lower_bound=0)
flags.DEFINE_integer("seed", 0, "rand seed")
flags.DEFINE_boolean("norm", True, "whether or not normalise gp's output")
flags.DEFINE_boolean("minimize", False, "min or max objective")

# flags cannot define dict, acq_func_config will be affected by side-effect
acq_func_config = {
    "q": 1,
    "num_restarts": 48,
    "raw_samples": 128,
    "num_samples": int(1024 * 2),
    # only a placeholder for {EI, qEI}, will be overwritten per iter
    "y_max": torch.tensor([1.]),
    "beta": 1,  # for UCB
}
train_inputs_dict = {}
train_targets_dict = {}

class n_dim_Rosenbrock(Metric):
    def fetch_trial_data(self, trial, **kwargs):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters

            # exec
            obj = call_rosenbrock(params, train_inputs_dict,
                                  train_targets_dict)
            mean = float(obj["final"])
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": mean,
                "sem": 0,
                "trial_index": trial.index,
            })
            print()
            print(f"trial: {trial.index} - reward: {mean:.2f}")
            print()
        return ax.core.data.Data(df=pd.DataFrame.from_records(records))

def main(_):

    # seeding
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    # load
    register_metric(n_dim_Rosenbrock)
    exp = load_exp(FLAGS.load_name)
    global train_inputs_dict, train_targets_dict
    train_inputs_dict, train_targets_dict = load_dict(FLAGS.load_name)

    print()
    print(f"==== resume from experiment sobol ====")
    print(exp.fetch_data().df)
    print()

    # get dag's spec
    param_space, metric_space, obj_space, edges = parse_model(
        FLAGS.performance_model_path)

    print()
    print(
        f"==== start experiment: {exp.name} with tuner: {FLAGS.tuner} & {FLAGS.acq_name} ===="
    )
    print()
    for t in range(FLAGS.epochs):
        start = time.perf_counter()

        model = build_model(FLAGS.tuner, exp, train_inputs_dict,
                            train_targets_dict, param_space, metric_space,
                            obj_space, edges, FLAGS.norm)
        candidates = inner_loop(
            exp,
            model,
            param_space,
            obj_space,
            edges,
            acq_name=FLAGS.acq_name,
            acq_func_config=acq_func_config,
        )

        # run
        gen_run = candidates_to_generator_run(exp, candidates)
        if acq_func_config["q"] == 1:
            trial = exp.new_trial(generator_run=gen_run)
        else:
            trial = exp.new_batch_trial(generator_run=gen_run)
        trial.run()
        trial.mark_completed()

        print()
        print("iteration time:")
        end = time.perf_counter() - start
        print(f"{end:.2f}")
        print()

        # update acq_func_config, e.g. update the best obs for expected improvement
        acq_func_config["y_max"] = train_targets_dict["final"].max()

    print()
    print(f"==== done experiment: {exp.name}====")
    print(print_experiment_result(exp))
    save_name = f"{FLAGS.exp_name}-{FLAGS.tuner}-{FLAGS.acq_name}"
    save_exp(exp, save_name)
    save_dict([train_inputs_dict, train_targets_dict, acq_func_config],
              save_name)


if __name__ == "__main__":
    app.run(main)
