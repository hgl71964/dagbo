import datetime
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

from dagbo.dag import Dag
from dagbo.fit_dag import fit_dag
from dagbo.utils.perf_model_utils import build_perf_model_from_spec_ssa, build_perf_model_from_spec_direct
from dagbo.utils.ax_experiment_utils import (candidates_to_generator_run,
                                             load_exp, get_dict_tensor,
                                             load_dict,
                                             print_experiment_result,
                                             save_dict, save_exp)
from dagbo.other_opt.bo_utils import get_fitted_model, inner_loop
from dagbo.interface.exec_spark import call_spark
from dagbo.interface.parse_performance_model import parse_model
from dagbo.interface.metrics_extractor import extract_throughput, extract_app_id, request_history_server
"""
load an experiment with initial sobol points & run opt loop
"""

FLAGS = flags.FLAGS
flags.DEFINE_enum("tuner", "dagbo", ["dagbo", "bo"], "tuner to use")
flags.DEFINE_string("exp_name", "SOBOL-spark-wordcount", "Experiment name")
flags.DEFINE_string("acq_name", "qEI", "acquisition function name")
flags.DEFINE_string("performance_model_path",
                    "dagbo/interface/rosenbrock_3d.txt",
                    "graphviz source path")

flags.DEFINE_integer("epochs", 20, "bo loop epoch", lower_bound=0)
flags.DEFINE_boolean("minimize", False, "min or max objective")

# flags cannot define dict, acq_func_config will be affected by side-effect
acq_func_config = {
    "q": 1,
    "num_restarts": 48,
    "raw_samples": 128,
    "num_samples": int(1024 * 2),
    "y_max": torch.tensor([
        1.
    ]),  # only a placeholder for {EI, qEI}, will be overwritten per iter
    "beta": 1,  # for UCB
}
torch_dtype = torch.float64
train_targets_dict = {}
normal_dict = {}
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
                "i1": torch.tensor(i1, dttorch_dtype),
                "i2": torch.tensor(i2, dttorch_dtype),
                "f1": torch.tensor(f1, dttorch_dtype),
                "f2": torch.tensor(f2, dttorch_dtype),
                "final": torch.tensor(final, dttorch_dtype),
            }

            for k, v in obj.items():

                val = v / normal_dict[k]  # XXX what if divide by 0?

                if k in train_targets_dict:
                    train_targets_dict[k] = torch.cat(
                        [train_targets_dict[k], val])
                else:
                    train_targets_dict[k] = val

            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": -float(train_targets_dict["final"][-1]),  # flip sign
                "sem": 0,  # 0 for noiseless experiment
                "trial_index": trial.index,
            })
            print()
            print(f"trial: {trial.index} - reward: {normalised_reward:.2f}x")
            print()
        return ax.core.data.Data(df=pd.DataFrame.from_records(records))


def get_model(exp: Experiment, param_names: list[str], param_space: dict,
              metric_space: dict, obj_space: dict, edges: dict,
              dtype) -> Union[Dag, SingleTaskGP]:

    # update acq_func_config, e.g. update the best obs for expected improvement
    acq_func_config["y_max"] = train_targets_dict["final"].max()

    if FLAGS.tuner == "bo":
        return get_fitted_model(exp, param_names, dtype)
    elif FLAGS.tuner == "dagbo":
        # input params can be read from ax experiment (`from scratch`)
        train_inputs_dict = get_dict_tensor(exp, param_names, dtype)

        ## fit model from dataset
        #build_perf_model_from_spec_direct, build_perf_model_from_spec_ssa
        dag = build_perf_model_from_spec_direct(train_inputs_dict,
                                                train_targets_dict,
                                                acq_func_config["num_samples"],
                                                param_space, metric_space,
                                                obj_space, edges)
        fit_dag(dag)
        return dag
    else:
        raise ValueError("unable to recognize tuner")


def main(_):
    register_metric(Rosenbrock_3D)
    exp = load_exp(FLAGS.exp_name)
    global train_targets_dict, normal_dict  # to change global var inside func
    train_targets_dict, normal_dict = load_dict(FLAGS.exp_name)
    print()
    print(f"==== resume from experiment sobol ====")
    print(exp.fetch_data().df)
    print("normalizer: ")
    print(normal_dict)
    print()

    # get dag's spec
    param_space, metric_space, obj_space, edges = parse_model(
        FLAGS.performance_model_path)

    # NOTE: ensure its EXACTLY the same as define in spark_sobol
    param_names = [
        "x1",
        "x2",
        "x3",
    ]

    print()
    print(
        f"==== start experiment: {exp.name} with tuner: {FLAGS.tuner} & {FLAGS.acq_name} ===="
    )
    print()
    for t in range(FLAGS.epochs):
        model = get_model(exp, param_names, param_space, metric_space,
                          obj_space, edges, torch_dtype)

        candidates = inner_loop(exp,
                                model,
                                param_names,
                                acq_name=FLAGS.acq_name,
                                acq_func_config=acq_func_config,
                                dtype=torch_dtype)
        gen_run = candidates_to_generator_run(exp, candidates, param_names)

        # run
        if acq_func_config["q"] == 1:
            trial = exp.new_trial(generator_run=gen_run)
        else:
            trial = exp.new_batch_trial(generator_run=gen_run)
        trial.run()
        trial.mark_completed()

    print()
    print(f"==== done experiment: {exp.name}====")
    print(print_experiment_result(exp))
    save_name = f"{FLAGS.exp_name}-{FLAGS.tuner}-{FLAGS.acq_name}"
    save_exp(exp, save_name)
    save_dict([train_targets_dict, acq_func_config], save_name)


if __name__ == "__main__":
    app.run(main)
