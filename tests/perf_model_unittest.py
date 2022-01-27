from absl import app
from absl import flags
from dagbo.utils.perf_model_utlis import *
from dagbo.interface.parse_performance_model import parse_model

FLAGS = flags.FLAGS
flags.DEFINE_string("performance_model_path", "dagbo/interface/spark_performance_model.txt", "graphviz source path")

def main(_):
    param_space, metric_space, obj_space, edges = parse_model(FLAGS.performance_model_path)
    print(param_space)
    print(edges)

if __name__ == "__main__":
    app.run(main)
