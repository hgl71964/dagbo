from absl import app
from absl import flags

import numpy as np
from hyperopt import fmin

from ax import Metric
from ax.storage.metric_registry import register_metric

from dagbo.interface.exec_spark import call_spark
from dagbo.utils.ax_experiment_utils import load_exp, save_dict, load_dict
from dagbo.utils.hyperopt_utils import search_space_from_ax_experiment, build_trials_from_sobol, get_model
from dagbo.interface.metrics_extractor import extract_and_aggregate, extract_duration_app_id, extract_throughput

FLAGS = flags.FLAGS
flags.DEFINE_enum("tuner", "rand", ["rand", "tpe"], "tuner to use")
flags.DEFINE_string("exp_name", "SOBOL-spark-wordcount", "Experiment name")
flags.DEFINE_string("load_name", "must provide", "load from experiment name")

flags.DEFINE_string("performance_model_path", "...", "graphviz source path")
flags.DEFINE_string(
    "conf_path", "/home/gh512/workspace/bo/spark-dir/hiBench/conf/spark.conf",
    "conf file path")
flags.DEFINE_string(
    "hibench_report_path", "must given",
    "hibench_report_path")
flags.DEFINE_string(
    "exec_path",
    "/home/gh512/workspace/bo/spark-dir/hiBench/bin/workloads/micro/wordcount/spark/run.sh",
    "executable path")
flags.DEFINE_string("base_url", "http://localhost:18080",
                    "history server base url")

flags.DEFINE_integer("epochs", 20, "opt loop epoch", lower_bound=0)
flags.DEFINE_integer("seed", 0, "rand seed")
flags.DEFINE_integer("minimize", 1, "min or max objective")


class SparkMetric(Metric):
    def fetch_trial_data(self, trial, **kwargs):
        pass


def obj(params: dict[str, float]) -> float:
    """
    e.g.
        x, y = params['x'], params['y']
        return np.sin(np.sqrt(x**2 + y**2))
    """
    # exec spark & retrieve throughput
    call_spark(params, FLAGS.conf_path, FLAGS.exec_path)
    #_, val = extract_duration_app_id(FLAGS.base_url)
    val, _ = extract_throughput(FLAGS.hibench_report_path)
    val = float(val)
    print()
    print(f"reward: {val:.2f}")
    print()
    # NOTE: convert if not minimize
    if bool(FLAGS.minimize):
        return val
    else:
        return -val


def main(_):
    # seeding
    np.random.seed(FLAGS.seed)

    # load
    register_metric(SparkMetric)
    exp = load_exp(FLAGS.load_name)
    #_, train_targets_dict = load_dict(FLAGS.load_name)

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
        algo=get_model(FLAGS.tuner),
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
