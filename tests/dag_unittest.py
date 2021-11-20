import os
import sys
import warnings
import logging
import unittest
import torch
from typing import List
from torch import Size, Tensor

# hacky way to include the src code dir
testdir = os.path.dirname(__file__)
srcdir = "../src"
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

# import from src
from dagbo.dag import Dag
from dagbo.dag_gpytorch_model import DagGPyTorchModel


class SIMPLE_DAG(Dag, DagGPyTorchModel):
    """
    creation a simple tree-like DAG

    x1      x2        x3
      \     /         |
        z1           z2
          \        /
              y
    """
    def __init__(self, train_input_names: List[str],
                 train_target_names: List[str], train_inputs: Tensor,
                 train_targets: Tensor, num_samples: int):
        super().__init__(train_input_names, train_target_names, train_inputs,
                         train_targets)

        # required for all classes that extend SparkDag
        self.num_samples = num_samples

    def define_dag(self, batch_shape: Size = Size([])) -> None:
        x_1 = self.register_input("x1")
        x_2 = self.register_input("x2")
        x_3 = self.register_input("x3")

        z_1 = self.register_metric("z1", [x_1, x_2])
        z_2 = self.register_metric("z2", [x_3])

        y = self.register_metric("y", [z_1, z_2])


class dag_test(unittest.TestCase):
    def setUp(self):
        """
        setUp is called before each test
        """
        #warnings.filterwarnings(action="ignore",
        #                        message="unclosed",
        #                        category=ResourceWarning)
        train_input_names = [
            "x1",
            "x2",
            "x3",
        ]
        train_target_names = [
            "z1",
            "z2",
            "y",
        ]
        batch_size = 1
        num_samples = 1000
        train_inputs = torch.rand(batch_size, 3, len(train_input_names))
        train_targets = torch.rand(batch_size, 3, len(train_target_names))

        self.simple_dag = SIMPLE_DAG(train_input_names, train_target_names,
                                     train_inputs, train_targets, num_samples)

    def tearDown(self):
        # gc
        pass

    @unittest.skip("no need")
    def test_sample(self):
        #logger.info("sample test")
        print("sample test that is skipped")

    #@unittest.skip("ok")
    def test_dag_creation(self):
        #print(self.simple_dag)
        for node in self.simple_dag.nodes_dag_order():
            print(node.output_name)

        for k, v in self.simple_dag.train_inputs_name2tensor_mapping.items():
            print(k)
            print(v.shape)

        print("stack list")
        stack_name = ["x1", "x2"]
        print(
            torch.stack([
                self.simple_dag.train_inputs_name2tensor_mapping[i]
                for i in stack_name
            ],
                        dim=-1).shape)

    def test_dag_forward(self):
        a = torch.stack([
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6]),
        ])
        print(a)

        b = torch.rand(1, 2)
        c = torch.ones(1, 2)

        print(torch.stack([b, c]).shape)

    @unittest.skip("..")
    def test_dag_backward(self):
        pass


if __name__ == '__main__':

    #logger = logging.getLogger()
    #logger.setLevel(logging.INFO)
    #handler = logging.StreamHandler()

    ## create formatter and add it to the handler
    #formatter = logging.Formatter(
    #    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #handler.setFormatter(formatter)
    ## add the handler to the logger
    #logger.addHandler(handler)

    unittest.main()
