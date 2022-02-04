from absl import app
from absl import flags

import ax
from ax.storage.metric_registry import register_metric

from dagbo.utils.ax_experiment_utils import load_exp

FLAGS = flags.FLAGS
flags.DEFINE_string("name",
                    "nbo",
                    "name of experiment")

class SparkMetric(ax.Metric):
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
    register_metric(SparkMetric)
    exp = load_exp(FLAGS.name)

    # TODO add visualisation?
    print(exp.fetch_data().df)
    print(exp.arms_by_name)

if __name__ == "__main__":
    app.run(main)
