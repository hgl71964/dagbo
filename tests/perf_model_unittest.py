import  unittest
from dagbo.utils.perf_model_utils import *
from dagbo.interface.parse_performance_model import parse_model
from dagbo.interface.metrics_extractor import request_history_server

class perf_utils_test(unittest.TestCase):
    def setUp(self):
        param_space, metric_space, obj_space, edges = parse_model("dagbo/interface/spark_performance_model.txt")
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

    #@unittest.skip("ok")
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

class perf_model_test(unittest.TestCase):

    def setUp(self):
        # performance model
        param_space, metric_space, obj_space, edges = parse_model("dagbo/interface/spark_performance_model.txt")
        self.param_space = param_space
        self.metric_space = metric_space
        self.obj_space = obj_space
        self.edges = edges
        #print(param_space)
        #print(edges)

        acq_func_config = {
            "q": 2,
            "num_restarts": 48,
            "raw_samples": 128,
            "num_samples": 2048,
            "y_max": torch.tensor([1.]),  # for EI
            "beta": 1,  # for UCB
        }
        self.acq_func_config = acq_func_config

        # make fake input tensor
        train_inputs_dict = {i: torch.rand(acq_func_config["q"]) for i in list(param_space.keys())}
        train_targets_dict = {i: torch.rand(acq_func_config["q"]) for i in list(metric_space.keys()) + list(obj_space.keys())}

        # build
        self.dag = build_perf_model_from_spec(train_inputs_dict,
                                   train_targets_dict,
                                   acq_func_config["num_samples"],
                                   param_space,
                                   metric_space,
                                   obj_space,
                                   edges)
        # feature extractor
        self.app_id = "application_1641844906451_0006"
        self.base_url = "http://localhost:18080"

    def test_feat_extract(self):
        metric = request_history_server(self.base_url, self.app_id)
        print(metric)

    def test_dag_build(self):
        print(self.dag)


if __name__ == "__main__":
    unittest.main()
