import os
import sys
from absl import app
from absl import flags
from typing import Union
import datetime

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

from dagbo.dag import Dag
from dagbo.fit_dag import fit_dag
from dagbo.utils.perf_model_utils import build_perf_model_from_spec_ssa, build_perf_model_from_spec_direct
from dagbo.utils.ax_experiment_utils import candidates_to_generator_run, save_exp, get_dict_tensor, save_dict, print_experiment_result
from dagbo.interface.exec_spark import call_spark
from dagbo.interface.parse_performance_model import parse_model
from dagbo.interface.metrics_extractor import extract_throughput, extract_app_id, request_history_server
"""
gen initial sobol points for an experiment
"""

FLAGS = flags.FLAGS
flags.DEFINE_string("metric_name", "rosenbrock", "metric name")
flags.DEFINE_string("exp_name", "rosenbrock-3D", "Experiment name")
flags.DEFINE_integer("bootstrap", 2, "bootstrap", lower_bound=1)
flags.DEFINE_boolean("minimize", False, "min or max objective")

train_targets_dict = {}
normal_dict = {}  # hold original value for metric, used for normalised metric
torch_dtype = torch.float64
np.random.seed(0)
torch.manual_seed(0)


class Rosenbrock_3D(Metric):
    def fetch_trial_data(self, trial, **kwargs):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters

            x1, x2, x3 = params["x1"], params["x2"], params["x3"]
            i1 = 100 * (x2 - x1**2)**2
            i2 = 100 * (x3 - x2**2)**2
            f1 = i1 + (1 - x1)**2
            f2 = i2 + (1 - x2)**2
            final = f1 + f2
            obj = {
                "i1": torch.tensor(i1, dtype=torch_dtype).reshape(-1),
                "i2": torch.tensor(i2, dtype=torch_dtype).reshape(-1),
                "f1": torch.tensor(f1, dtype=torch_dtype).reshape(-1),
                "f2": torch.tensor(f2, dtype=torch_dtype).reshape(-1),
                "final": torch.tensor(final, dtype=torch_dtype).reshape(-1),
            }

            for k, v in obj.items():
                if k not in normal_dict:
                    if k == "final":        # NOTE: flip sign
                        normal_dict[k] = -v
                    else:
                        normal_dict[k] = v

                val = v / normal_dict[k]  # XXX what if divide by 0?

                if k in train_targets_dict:
                    train_targets_dict[k] = torch.cat(
                        [train_targets_dict[k], val])
                else:
                    train_targets_dict[k] = val

            # the latest reward
            mean = float(train_targets_dict["final"][-1])
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": mean,
                "sem": 0,  # 0 for noiseless experiment
                "trial_index": trial.index,
            })
            print()
            print(f"trial: {trial.index} - reward: {mean:.2f}x")
            print()
        return ax.core.data.Data(df=pd.DataFrame.from_records(records))


def main(_):

    # for saving
    register_metric(Rosenbrock_3D)

    # build experiment
    ## for now need to define manually
    param_names = [
        "x1",
        "x2",
        "x3",
    ]
    search_space = SearchSpace([
        ax.RangeParameter("x1", ax.ParameterType.FLOAT, lower=0, upper=3),
        ax.RangeParameter("x2", ax.ParameterType.FLOAT, lower=0, upper=3),
        ax.RangeParameter("x3", ax.ParameterType.FLOAT, lower=0, upper=3),
    ])
    optimization_config = OptimizationConfig(
        Objective(metric=Rosenbrock_3D(name=FLAGS.metric_name),
                  minimize=FLAGS.minimize))
    exp = Experiment(name=FLAGS.exp_name,
                     search_space=search_space,
                     optimization_config=optimization_config,
                     runner=SyntheticRunner())

    print()
    print(f"==== start SOBOL experiment: {exp.name} ====")
    print()

    ## BOOTSTRAP
    sobol = Models.SOBOL(exp.search_space)
    generated_run = sobol.gen(FLAGS.bootstrap)
    trial = exp.new_batch_trial(generator_run=generated_run)
    trial.run()
    trial.mark_completed()

    print()
    print(f"==== done SOBOL experiment ====")
    print()

    print(print_experiment_result(exp))

    # save
    dt = datetime.datetime.today()
    save_name = f"SOBOL-{FLAGS.exp_name}"
    save_exp(exp, save_name)
    save_dict([train_targets_dict, normal_dict], save_name)


if __name__ == "__main__":
    app.run(main)
