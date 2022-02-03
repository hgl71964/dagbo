import os
import sys
from absl import app
from absl import flags

import pandas as pd
import torch
from torch import Tensor

import ax
from ax.modelbridge.registry import Models
from ax import SearchSpace, Experiment, OptimizationConfig, Objective, Metric
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.runners.synthetic import SyntheticRunner

from dagbo.fit_dag import fit_dag
from dagbo.utils.perf_model_utils import build_perf_model_from_spec
from dagbo.interface.exec_spark import call_spark
from dagbo.interface.parse_performance_model import parse_model
from dagbo.interface.metrics_extractor import extract_throughput, extract_app_id, request_history_server

FLAGS = flags.FLAGS
flags.DEFINE_string("performance_model_path",
                    "dagbo/interface/spark_performance_model.txt",
                    "graphviz source path")
flags.DEFINE_string("metric_name", "spark_throughput", "metric name")
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

flags.DEFINE_integer("epochs", 5, "bo loop epoch", lower_bound=0)
flags.DEFINE_integer("bootstrap", 3, "bootstrap", lower_bound=2)
flags.DEFINE_boolean("minimize", False, "min or max objective")

# flags cannot define dict
acq_func_config = {
    "q": 1,
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

            # exec spark & retrieve throughput
            call_spark(params, FLAGS.conf_path, FLAGS.exec_path)
            val = extract_throughput(FLAGS.hibench_report_path)

            # to records
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": float(val),
                "sem": 0,  # 0 for noiseless experiment
                "trial_index": trial.index,
            })
        return ax.core.data.Data(df=pd.DataFrame.from_records(records))


def main(_):

    # build experiment
    ## get dag's spec
    param_space, metric_space, obj_space, edges = parse_model(
        FLAGS.performance_model_path)

    ## for now need to define manually
    param_names = [
        "executor.num[*]",
        "executor.cores",
        "shuffle.compress",
        "memory.fraction",
        "executor.memory",
        "spark.serializer",
        "rdd.compress",
        "default.parallelism",
        "shuffle.spill.compress",
        "spark.speculation",
    ]
    search_space = SearchSpace([
        ax.RangeParameter("executor.num[*]",
                          ax.ParameterType.FLOAT,
                          lower=2,
                          upper=8),
        ax.RangeParameter("executor.cores",
                          ax.ParameterType.FLOAT,
                          lower=1,
                          upper=4),
        ax.RangeParameter("shuffle.compress",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
        ax.RangeParameter("memory.fraction",
                          ax.ParameterType.FLOAT,
                          lower=0.1,
                          upper=0.99),
        ax.RangeParameter("executor.memory",
                          ax.ParameterType.FLOAT,
                          lower=2,
                          upper=4),
        ax.RangeParameter("spark.serializer",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
        ax.RangeParameter("rdd.compress",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
        ax.RangeParameter("default.parallelism",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
        ax.RangeParameter("shuffle.spill.compress",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
        ax.RangeParameter("spark.speculation",
                          ax.ParameterType.FLOAT,
                          lower=0,
                          upper=1),
    ])
    optimization_config = OptimizationConfig(
        Objective(metric=SparkMetric(name=FLAGS.metric_name),
                  minimize=FLAGS.minimize))
    exp = Experiment(name="spark_feed_back_loop",
                     search_space=search_space,
                     optimization_config=optimization_config,
                     runner=SyntheticRunner())

    ## BOOTSTRAP
    sobol = Models.SOBOL(exp.search_space)
    generated_run = sobol.gen(FLAGS.bootstrap)
    trial = exp.new_batch_trial(generator_run=generated_run)
    trial.run()
    trial.mark_completed()

    print(exp.fetch_data().df)
    raise RuntimeError()

    # bo loop
    for t in range(FLAGS.epochs):
        # input params can be read from ax experiment
        train_inputs_dict = get_dict_tensor(exp, param_names)

        # TODO
        train_targets_dict = {}

        # fit model from dataset
        dag = build_perf_model_from_spec(train_inputs_dict, train_targets_dict,
                                         acq_func_config["num_samples"],
                                         param_space, metric_space, obj_space,
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

        # TODO
        app_id = extract_app_id(FLAGS.log_path)
        metric = request_history_server(FLAGS.base_url, app_id)

    print("done")
    print(exp.fetch_data().df)


if __name__ == "__main__":
    app.run(main)
