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

from dagbo.dag import Dag
from dagbo.fit_dag import fit_dag
from dagbo.utils.ax_experiment_utils import save_exp, save_dict, print_experiment_result
from dagbo.interface.exec_spark import call_spark
from dagbo.interface.parse_performance_model import parse_model
from dagbo.interface.metrics_extractor import extract_throughput, extract_app_id, request_history_server
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
flags.DEFINE_string(
    "exec_path",
    "/home/gh512/workspace/bo/spark-dir/hiBench/bin/workloads/micro/wordcount/spark/run.sh",
    "executable path")
flags.DEFINE_string(
    "log_path",
    "/home/gh512/workspace/bo/spark-dir/hiBench/report/wordcount/spark/bench.log",
    "log file's path for app id extraction")
flags.DEFINE_string(
    "hibench_report_path",
    "/home/gh512/workspace/bo/spark-dir/hiBench/report/hibench.report",
    "hibench report file path")
flags.DEFINE_string("base_url", "http://localhost:18080",
                    "history server base url")

train_inputs_dict = {}
train_targets_dict = {}


class SparkMetric(Metric):
    def fetch_trial_data(self, trial, **kwargs):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters

            # TODO
            # exec spark & retrieve throughput
            call_spark(params, FLAGS.conf_path, FLAGS.exec_path)
            val = extract_throughput(FLAGS.hibench_report_path)

            # extract and append intermediate metric
            app_id = extract_app_id(FLAGS.log_path)
            metric = request_history_server(FLAGS.base_url, app_id)

            ## get metrics across executors
            agg_m = {}
            for _, perf in metric.items():
                for monitoring_metic, v in perf.items():
                    if monitoring_metic in agg_m:
                        agg_m[monitoring_metic].append(
                            float(v))  # XXX all monitoring v are float?
                    else:
                        agg_m[monitoring_metic] = [float(v)]
            ### add final obj
            agg_m["throughput"] = float(val)

            ## aggregate & normalised metrics
            for k, v in agg_m.items():
                # convert to tensor & average
                agg_v = torch.tensor(v, dtype=torch_dtype).mean().reshape(-1)

                # use the first occurrence val as the normalizer
                if k not in normal_dict:
                    if agg_v == 0:  # XXX 0 as the normalizer
                        normal_dict[k] = torch.tensor([1.], dtype=torch_dtype)
                        agg_m[k] = agg_v
                    else:
                        normal_dict[k] = agg_v
                        agg_m[k] = torch.tensor([1.], dtype=torch_dtype)

                else:
                    agg_m[k] = agg_v / normal_dict[k]

            ## populate
            for k, v in agg_m.items():
                if k in train_targets_dict:
                    train_targets_dict[k] = torch.cat(
                        [train_targets_dict[k], v])
                else:
                    train_targets_dict[k] = v

            # to records
            normalised_reward = float(agg_m["throughput"])
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": normalised_reward,
                "sem": 0,  # 0 for noiseless experiment
                "trial_index": trial.index,
            })
            print()
            print(f"trial: {trial.index} - reward: {normalised_reward:.2f}x")
            print()
        return ax.core.data.Data(df=pd.DataFrame.from_records(records))


def main(_):

    # seeding
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    # build experiment
    ## for now need to define manually
    param_names = [
        "executor.num[*]",
        "executor.cores",
        "memory.fraction",
        "executor.memory",
        "default.parallelism",
    ]
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
    ])
    optimization_config = OptimizationConfig(
        Objective(metric=SparkMetric(name=FLAGS.metric_name),
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
    register_metric(SparkMetric)
    save_name = f"SOBOL-{FLAGS.exp_name}"
    save_exp(exp, save_name)
    save_dict([train_targets_dict, normal_dict], save_name)


if __name__ == "__main__":
    app.run(main)
