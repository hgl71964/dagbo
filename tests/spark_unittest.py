import ax
import unittest
import time
from time import sleep
import numpy as np

from dagbo.interface.exec_spark import *
from dagbo.interface.metrics_extractor import *
from dagbo.interface.parse_performance_model import parse_model
from dagbo.utils.perf_model_utils import find_inverse_edges, get_dag_topological_order


class exec_spark_test(unittest.TestCase):
    def setUp(self):
        self.app_id = "application_1646152825604_5728"
        self.base_url = "http://localhost:18080"

    def tearDown(self):
        pass

    @unittest.skip("ok")
    def test_feat_extract(self):
        metric = request_history_server(self.base_url, self.app_id)
        print(metric)

    @unittest.skip("ok")
    def test_extract_throughput(self):
        path = "/home/gh512/workspace/bo/spark-dir/hiBench/report/hibench.report"
        l = extract_throughput(path)
        print(l)

    #@unittest.skip("ok")
    def test_app_id_feat_extraction(self):
        log_path = "/home/gh512/workspace/bo/spark-dir/hiBench/report/wordcount/spark/bench.log"
        app_id, _ = extract_duration_app_id(self.base_url)
        metric = request_history_server(self.base_url, app_id)

        print("end-to-end feat extraction: ")
        print(metric[0])

    @unittest.skip("ok")
    def test_subprocess(self):
        rc = subprocess.run(["ls", "-l"])
        print(rc)
        print("...")

        t = time.time()
        rc = subprocess.run(["sleep", "3"])
        print(rc)
        print(time.time() - t)

        if rc.returncode != 0:
            print("return != 0")

    @unittest.skip("ok, this will overwrite conf file")
    def test_call_spark(self):
        conf_path = "/home/gh512/workspace/bo/spark-dir/hiBench/conf/spark.conf"
        exec_path = "/home/gh512/workspace/bo/spark-dir/hiBench/bin/workloads/micro/wordcount/spark/run.sh"

        # spec as generated by bo
        spec = {
            "executor.num[*]": 2,
            "executor.cores": 1,
            "executor.memory": 2,
            "shuffle.compress": 0,
        }
        call_spark(spec, conf_path, exec_path)


class perf_utils_test(unittest.TestCase):
    def setUp(self):
        param_space, metric_space, obj_space, edges = parse_model(
            "dagbo/interface/rosenbrock_20d_bo.txt")
        #"dagbo/interface/spark_performance_model.txt")

        self.param_space = param_space
        self.metric_space = metric_space
        self.obj_space = obj_space
        self.edges = edges
        #print(param_space)
        #print(edges)

    @unittest.skip("ok")
    def test_reversed_edge(self):
        reversed_edge = find_inverse_edges(self.edges)
        print(reversed_edge)

    @unittest.skip("ok")
    def test_topological_sort(self):
        order = get_dag_topological_order(self.obj_space, self.edges)
        print(order)
        """
        A topological sort of a dag G = (V,E) is a linear ordering of all its vertices such that if G contains an edge (u,v),
            then u appears before v in the ordering.
        """
        for key in self.edges:
            for val in self.edges[key]:
                for node in order:
                    if key == node:  # find key first, ok
                        break
                    elif val == node:
                        raise RuntimeError("not topological order")


if __name__ == "__main__":
    unittest.main()
