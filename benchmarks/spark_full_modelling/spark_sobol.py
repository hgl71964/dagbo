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

from dagbo.utils.ax_experiment_utils import save_exp, save_dict, print_experiment_result
from dagbo.interface.exec_spark import call_spark
from dagbo.interface.metrics_extractor import extract_and_aggregate
"""
gen initial sobol points for an experiment
"""

FLAGS = flags.FLAGS
flags.DEFINE_string("metric_name", "spark_throughput", "metric name")
flags.DEFINE_string("exp_name", "spark-wordcount", "Experiment name")
flags.DEFINE_integer("bootstrap", 5, "bootstrap", lower_bound=1)
flags.DEFINE_integer("seed", 0, "rand seed")
flags.DEFINE_integer("minimize", 0, "min or max objective")

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

# global var so that SparkMetric can populate
train_inputs_dict = {}
train_targets_dict = {}


class SparkMetric(Metric):
    def fetch_trial_data(self, trial, **kwargs):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters

            # exec spark & retrieve throughput
            call_spark(params, FLAGS.conf_path, FLAGS.exec_path)
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

    # build experiment
    ## for now need to define manually
    search_space = SearchSpace([
        ax.RangeParameter("executor.num[*]",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
        ax.RangeParameter("executor.cores",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
        ax.RangeParameter("memory.fraction",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
        ax.RangeParameter("executor.memory",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
        ax.RangeParameter("default.parallelism",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
        ax.RangeParameter("spark.shuffle.file.buffer",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
        ax.RangeParameter("spark.speculation.multiplier",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
        ax.RangeParameter("spark.speculation.quantile",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
        ax.RangeParameter("spark.broadcast.blockSize",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
        ax.RangeParameter("spark.kryoserializer.buffer",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
    ])
    optimization_config = OptimizationConfig(
        Objective(metric=SparkMetric(name=FLAGS.metric_name),
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
    register_metric(SparkMetric)
    save_name = f"SOBOL-{FLAGS.exp_name}"
    save_exp(exp, save_name)
    save_dict([train_inputs_dict, train_targets_dict], save_name)


if __name__ == "__main__":
    app.run(main)
