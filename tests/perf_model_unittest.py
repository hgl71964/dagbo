import  unittest
from dagbo.utils.perf_model_utlis import *
from dagbo.interface.parse_performance_model import parse_model

class perf_model_test(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
