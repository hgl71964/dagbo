import datetime
from absl import app
from absl import flags
from typing import Union

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
                                             load_train_targets_dict,
                                             print_experiment_result, save_exp)
from dagbo.other_opt.bo_utils import get_fitted_model, inner_loop
from dagbo.interface.exec_spark import call_spark
from dagbo.interface.parse_performance_model import parse_model
from dagbo.interface.metrics_extractor import extract_throughput, extract_app_id, request_history_server
"""
load an experiment with initial sobol points & run opt loop
"""

FLAGS = flags.FLAGS
flags.DEFINE_enum("tuner", "dagbo", ["dagbo", "bo"], "tuner to use")
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

flags.DEFINE_integer("epochs", 20, "bo loop epoch", lower_bound=0)
flags.DEFINE_boolean("minimize", False, "min or max objective")

# flags cannot define dict
acq_func_config = {
    "q": 1,
    "num_restarts": 48,
    "raw_samples": 128,
    "num_samples": int(1024 * 2),
    "y_max": torch.tensor([1.]),  # for EI
    "beta": 1,  # for UCB
}
exp_name = "SOBOL-spark_feed_back_loop-2022-2-10"
acq_name = "qUCB"
torch_dtype = torch.float64


class SparkMetric(Metric):
    def fetch_trial_data(self, trial, **kwargs):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters

            # exec spark & retrieve throughput
            call_spark(params, FLAGS.conf_path, FLAGS.exec_path)
            val = extract_throughput(FLAGS.hibench_report_path)

            # extract and append intermediate metric
            app_id = extract_app_id(FLAGS.log_path)
            metric = request_history_server(FLAGS.base_url, app_id)

            ## average agg
            agg_m = {}
            for _, perf in metric.items():
                for monitoring_metic, v in perf.items():
                    if monitoring_metic in agg_m:
                        agg_m[monitoring_metic].append(
                            float(v))  # XXX all monitoring v are float?
                    else:
                        agg_m[monitoring_metic] = [float(v)]

            for k, v in agg_m.items():
                agg_m[k] = torch.tensor(v, dtype=torch_dtype).mean().reshape(
                    -1)  # convert to tensor & average
            agg_m["throughput"] = torch.tensor(float(val),
                                               dtype=torch_dtype).reshape(-1)

            ### populate
            for k, v in agg_m.items():
                if k in train_targets_dict:
                    train_targets_dict[k] = torch.cat(
                        [train_targets_dict[k], v])  # float32
                else:
                    train_targets_dict[k] = v

            # to records
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": float(val),
                "sem": 0,  # 0 for noiseless experiment
                "trial_index": trial.index,
            })

            print()
            print(f"trial: {trial.index} - reward: {val}")
            print()

        return ax.core.data.Data(df=pd.DataFrame.from_records(records))


def get_model(exp: Experiment, param_names: list[str], param_space: dict,
              metric_space: dict, obj_space: dict, edges: dict,
              dtype) -> Union[Dag, SingleTaskGP]:
    if FLAGS.tuner == "bo":
        return get_fitted_model(exp, param_names, dtype)
    elif FLAGS.tuner == "dagbo":
        # input params can be read from ax experiment (`from scratch`)
        train_inputs_dict = get_dict_tensor(exp, param_names, dtype)

        ## fit model from dataset
        #build_perf_model_from_spec_direct, build_perf_model_from_spec_ssa
        dag = build_perf_model_from_spec_ssa(train_inputs_dict,
                                             train_targets_dict,
                                             acq_func_config["num_samples"],
                                             param_space, metric_space,
                                             obj_space, edges)
        fit_dag(dag)
        return dag
    else:
        raise ValueError("unable to recognize tuner")


register_metric(SparkMetric)
exp = load_exp(exp_name)
train_targets_dict = load_train_targets_dict(exp_name)


def main(_):
    # get dag's spec
    param_space, metric_space, obj_space, edges = parse_model(
        FLAGS.performance_model_path)

    # NOTE: ensure its the same as define in spark_sobol
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

    print()
    print(f"==== start experiment: {exp.name} with tuner: {FLAGS.tuner} ====")
    print()
    for t in range(FLAGS.epochs):
        model = get_model(exp, param_names, param_space, metric_space,
                          obj_space, edges, torch_dtype)

        candidates = inner_loop(exp,
                                model,
                                param_names,
                                acq_name=acq_name,
                                acq_func_config=acq_func_config,
                                dtype=torch_dtype)
        gen_run = candidates_to_generator_run(exp, candidates, param_names)

        # before run, param will be type-checked, so some XXX param needs to be conversed before this line
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
    dt = datetime.datetime.today()
    save_exp(
        exp,
        f"{exp.name}-{FLAGS.tuner}-{acq_name}-{dt.year}-{dt.month}-{dt.day}")
    # TODO save other exp param as well?


if __name__ == "__main__":
    app.run(main)
