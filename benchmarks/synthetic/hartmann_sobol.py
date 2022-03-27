from absl import app
from absl import flags

import numpy as np
import pandas as pd
import torch

import ax
from ax.modelbridge.registry import Models
from ax import SearchSpace, Experiment, OptimizationConfig, Objective, Metric
from ax.storage.metric_registry import register_metric
from ax.runners.synthetic import SyntheticRunner

from dagbo.interface.exec_hartmann import call_hartmann
from dagbo.utils.ax_experiment_utils import save_exp, save_dict, print_experiment_result
"""
gen initial sobol points for an experiment
"""

FLAGS = flags.FLAGS
flags.DEFINE_string("metric_name", "hartmann", "metric name")
flags.DEFINE_string("exp_name", "hartmann_6D", "Experiment name")
flags.DEFINE_integer("bootstrap", 5, "bootstrap", lower_bound=1)
flags.DEFINE_integer("n_dim", 6, "6-dim only")
flags.DEFINE_integer("seed", 0, "rand seed")
flags.DEFINE_integer("minimize", 0, "min or max objective")
train_inputs_dict = {}
train_targets_dict = {}


class n_dim_Rosenbrock(Metric):
    def fetch_trial_data(self, trial, **kwargs):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            # exec
            obj = call_hartmann(params, train_inputs_dict,
                                  train_targets_dict)
            mean = float(obj["final"])
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": mean,
                "sem": 0,
                "trial_index": trial.index,
            })
        return ax.core.data.Data(df=pd.DataFrame.from_records(records))


def main(_):

    # seeding
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    # build experiment
    param_names = [f"x{i}" for i in range(FLAGS.n_dim)]
    search_space = SearchSpace([
        ax.RangeParameter(i, ax.ParameterType.FLOAT, lower=0, upper=1)
        for i in param_names
    ])

    optimization_config = OptimizationConfig(
        Objective(metric=n_dim_Rosenbrock(name=FLAGS.metric_name),
                  minimize=bool(FLAGS.minimize)))
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
    register_metric(n_dim_Rosenbrock)
    save_name = f"SOBOL-{FLAGS.exp_name}"
    save_exp(exp, save_name)
    save_dict([train_inputs_dict, train_targets_dict], save_name)


if __name__ == "__main__":
    app.run(main)
