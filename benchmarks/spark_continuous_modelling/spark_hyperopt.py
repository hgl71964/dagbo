import datetime
from absl import app
from absl import flags

import numpy as np
from hyperopt import fmin, tpe, rand

from ax import SearchSpace, Experiment, OptimizationConfig, Objective, Metric
from ax.storage.metric_registry import register_metric

from dagbo.interface.exec_spark import call_spark
from dagbo.utils.ax_experiment_utils import load_exp, save_dict, load_dict
from dagbo.utils.hyperopt_utils import search_space_from_ax_experiment, build_trials_from_sobol
from dagbo.interface.metrics_extractor import extract_throughput

FLAGS = flags.FLAGS
flags.DEFINE_enum("tuner", "rand", ["rand", "tpe"], "tuner to use")
flags.DEFINE_string("exp_name", "SOBOL-spark-wordcount", "Experiment name")
flags.DEFINE_string("load_name", "SOBOL-spark-wordcount", "Experiment name")
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

train_targets_dict = {}
normal_dict = {}
#np.random.seed(0)


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


def get_model():
    if FLAGS.tuner == "rand":
        return rand.suggest
    elif FLAGS.tuner == "tpe":
        return tpe.suggest
    else:
        raise ValueError("unable to recognize tuner")


def obj(params: dict[str, float]) -> float:
    # exec spark & retrieve throughput
    call_spark(params, FLAGS.conf_path, FLAGS.exec_path)
    val = extract_throughput(FLAGS.hibench_report_path)

    # normalise reward
    normalised_val = float(val) / float(normal_dict["throughput"])

    print()
    print(f"reward: {normalised_val:.2f}x")
    print()

    # NOTE: convert to negative as hyperopt mins obj
    return -normalised_val


#def obj(params):
#    x, y = params['x'], params['y']
#    return np.sin(np.sqrt(x**2 + y**2))


def main(_):
    # load experiment
    register_metric(SparkMetric)
    exp = load_exp(FLAGS.load_name)
    global train_targets_dict, normal_dict  # to change global var inside func
    train_targets_dict, normal_dict = load_dict(FLAGS.load_name)

    print()
    print(f"==== resume from experiment sobol ====")
    print(exp.fetch_data().df)
    print()

    # build trials (hyperopt's opt history container)
    t = build_trials_from_sobol(exp)

    print()
    print(f"==== start experiment: {exp.name} with tuner: {FLAGS.tuner} ====")
    print()

    best = fmin(
        fn=obj,
        space=search_space_from_ax_experiment(exp),
        algo=get_model(),
        max_evals=FLAGS.epochs + len(t),  # total evals num
        trials=
        t,  # NOTE: sobols are passed as trials, t is updated by fmin side-effect
    )

    print()
    print(f"==== done experiment: {exp.name}====")
    print(best)
    save_name = f"{FLAGS.exp_name}-{FLAGS.tuner}"
    save_dict(t.trials, save_name)


if __name__ == "__main__":
    app.run(main)
