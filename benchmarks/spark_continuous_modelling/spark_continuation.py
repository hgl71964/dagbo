import time
from absl import app
from absl import flags

import numpy as np
import pandas as pd
import torch
import ax
from ax import Experiment, Metric
from ax.storage.metric_registry import register_metric

from dagbo.models.model_builder import build_model
from dagbo.acq_func.acq_func import inner_loop
from dagbo.utils.basic_utils import gpu_usage
from dagbo.utils.ax_experiment_utils import (candidates_to_generator_run,
                                             load_exp, load_dict,
                                             print_experiment_result,
                                             save_dict, save_exp)
from dagbo.interface.exec_spark import call_spark
from dagbo.interface.parse_performance_model import parse_model
from dagbo.interface.metrics_extractor import extract_and_aggregate
"""
load an experiment with initial sobol points & run opt loop
"""

FLAGS = flags.FLAGS
flags.DEFINE_enum("tuner", "dagbo-ssa", ["dagbo-direct", "dagbo-ssa", "bo"],
                  "tuner to use")
flags.DEFINE_enum("device", "gpu", ["cpu", "gpu"], "device to use")
flags.DEFINE_string("exp_name", "SOBOL-spark-wordcount", "Experiment name")
flags.DEFINE_string("load_name", "must provide", "load from experiment name")
flags.DEFINE_string("acq_name", "qUCB", "acquisition function name")
flags.DEFINE_string("performance_model_path", "must provide",
                    "graphviz source path")

flags.DEFINE_string(
    "conf_path", "/home/gh512/workspace/bo/spark-dir/hiBench/conf/spark.conf",
    "conf file path")
flags.DEFINE_string("hibench_report_path", "must given", "hibench_report_path")
flags.DEFINE_string(
    "exec_path",
    "/home/gh512/workspace/bo/spark-dir/hiBench/bin/workloads/micro/wordcount/spark/run.sh",
    "executable path")
flags.DEFINE_string("base_url", "http://localhost:18080",
                    "history server base url")

flags.DEFINE_integer("epochs", 20, "bo loop epoch", lower_bound=0)
flags.DEFINE_integer("seed", 0, "rand seed")
flags.DEFINE_integer("norm", 1, "whether or not normalise gp's output")
flags.DEFINE_integer("minimize", 1, "min or max objective")

# flags cannot define dict, acq_func_config will be affected by side-effect
acq_func_config = {
    "q": 1,
    "num_restarts": 128,  # 128
    "raw_samples": int(512),
    "num_samples": int(512),  # 1024
    "sbo_samples": int(512),  # 1024
    "y_max": torch.tensor([
        1.
    ]),  # only a placeholder for {EI, qEI}, will be overwritten per iter
    "beta": 10,  # for UCB
}
train_inputs_dict = {}
train_targets_dict = {}


class SparkMetric(Metric):
    def fetch_trial_data(self, trial, **kwargs):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            # exec spark & retrieve throughput
            call_spark(params, FLAGS.conf_path, FLAGS.exec_path)
            # side-effect
            val = extract_and_aggregate(params, train_inputs_dict,
                                        train_targets_dict,
                                        FLAGS.hibench_report_path,
                                        FLAGS.base_url, FLAGS.conf_path)
            # to records
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": val,
                "sem": 0,  # 0 for noiseless experiment
                "trial_index": trial.index,
            })
        return ax.core.data.Data(df=pd.DataFrame.from_records(records))


def main(_):
    # seeding
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    # load
    register_metric(SparkMetric)
    exp = load_exp(FLAGS.load_name)
    global train_inputs_dict, train_targets_dict
    train_inputs_dict, train_targets_dict = load_dict(FLAGS.load_name)

    # device
    device = torch.device("cpu")
    device_name = device
    if torch.cuda.is_available() and FLAGS.device == "gpu":
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name()

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
    print("minimize: ", bool(FLAGS.minimize))
    print()
    for t in range(FLAGS.epochs):
        start = time.perf_counter()

        model = build_model(FLAGS.tuner, train_inputs_dict,
                            train_targets_dict, param_space,
                            metric_space, obj_space, edges, acq_func_config,
                            bool(FLAGS.norm), bool(FLAGS.minimize), device)
        candidates = inner_loop(
            exp,
            model,
            param_space,
            acq_name=FLAGS.acq_name,
            acq_func_config=acq_func_config,
        )
        gen_run = candidates_to_generator_run(exp, candidates, param_space)

        # run
        # before run, param will be type-checked,
        # so some XXX param needs to be conversed before this line
        if acq_func_config["q"] == 1:
            trial = exp.new_trial(generator_run=gen_run)
        else:
            trial = exp.new_batch_trial(generator_run=gen_run)
        trial.run()
        trial.mark_completed()
        print()
        print(f"iteration {t+1}:")
        end = time.perf_counter() - start
        res = float(trial.fetch_data().df["mean"])
        print(f"time: {end:.2f} - results: {res:.2f}")
        if torch.cuda.is_available() and FLAGS.device == "gpu":
            torch.cuda.empty_cache()
            gpu_usage()
        print()

    print()
    print(f"==== done experiment: {exp.name}====")
    print(print_experiment_result(exp))
    save_name = f"{FLAGS.exp_name}-{FLAGS.tuner}-{FLAGS.acq_name}"
    save_exp(exp, save_name)
    save_dict([train_inputs_dict, train_targets_dict, acq_func_config],
              save_name)


if __name__ == "__main__":
    app.run(main)
