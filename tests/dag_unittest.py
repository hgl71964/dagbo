import os
import sys
import warnings
import logging
import unittest
import torch
import pandas as pd
from typing import List
from torch import Size, Tensor

# hacky way to include the src code dir
testdir = os.path.dirname(__file__)
srcdir = "../src"
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

# import from src
from dagbo.dag import Dag
from dagbo.dag_gpytorch_model import DagGPyTorchModel


class TREE_DAG(Dag, DagGPyTorchModel):
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


# original SBO implementation as in dagbo for spark
class original_dag_test(unittest.TestCase):
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
        train_inputs = 2*torch.rand(batch_size, 3, len(train_input_names))-1
        train_targets = 2*torch.rand(batch_size, 3, len(train_target_names))-1

        self.simple_dag = TREE_DAG(train_input_names, train_target_names,
                                   train_inputs, train_targets, num_samples)

    def tearDown(self):
        # gc
        pass

    @unittest.skip("this shows how a ax experiment is done")
    def test_ax_exp(self):
        import ax
        from ax import RangeParameter, ChoiceParameter, ParameterType, \
            SearchSpace, Experiment, OptimizationConfig, Objective, Metric
        from ax.modelbridge.registry import Models
        from dagbo.ax_utils import AxDagModelConstructor, register_runners

        # parameter space
        x1 = RangeParameter("x1", ParameterType.FLOAT, lower=-1, upper=1)
        x2 = RangeParameter("x2", ParameterType.FLOAT, lower=-1, upper=1)
        x3 = RangeParameter("x3", ParameterType.FLOAT, lower=-1, upper=1)
        z1 = RangeParameter("z1", ParameterType.FLOAT, lower=-1, upper=1)
        z2 = RangeParameter("z2", ParameterType.FLOAT, lower=-1, upper=1)
        y = RangeParameter("y", ParameterType.FLOAT, lower=-1, upper=1)
        parameters = [x1, x2, x3, z1, z2, y]
        search_space = SearchSpace(parameters)

        # obj config
        class CustomMetric(Metric):
            # must impl this method
            def fetch_trial_data(self, trial, **kwargs):
                records = []
                for arm_name, arm in trial.arms_by_name.items():
                    records.append({
                        "arm_name": arm_name,
                        "metric_name": self.name,
                        "mean":
                        0.0,  # mean value of this metric when this arm is used
                        "sem": 0.2,  # standard error of the above mean
                        "trial_index": trial.index,
                    })
                    print("from metric")
                    print(arm_name)
                    print(arm)
                return ax.core.data.Data(df=pd.DataFrame.from_records(records))

        optimization_config = OptimizationConfig(
            Objective(metric=CustomMetric(name="custom_obj"), minimize=True))

        # runners - control how experiment is deployed - mapping from arms to other APIs
        class dummy_runner(ax.Runner):
            def run(self, trial):
                trial_metadata = {"from_runner - name": str(trial.index)}
                print("dummy runner")
                print(trial.arms)
                print(trial.index)
                return trial_metadata

        register_runners()

        # experiment
        exp = Experiment(name="test_exp",
                         search_space=search_space,
                         optimization_config=optimization_config,
                         runner=dummy_runner())
        print(exp)

        # BOOTSTRAP EVALUATIONS
        num_bootstrap = 2
        sobol = Models.SOBOL(exp.search_space)
        trial = exp.new_batch_trial(generator_run=sobol.gen(num_bootstrap))
        trial.run()
        trial.mark_completed()

        # run BO
        epochs = 3
        for i in range(epochs):
            # Reinitialize GP+EI model at each step with updated data.
            gpei = Models.BOTORCH(experiment=exp, data=exp.fetch_data())
            generator_run = gpei.gen(n=1)
            trial = exp.new_trial(generator_run=generator_run)
            trial.run()
            trial.mark_completed()
        print("done")
        print(exp.fetch_data().df)


# SEM modelling
class SEM_dag_test(unittest.TestCase):
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

        self.simple_dag = TREE_DAG(train_input_names, train_target_names,
                                   train_inputs, train_targets, num_samples)

    def tearDown(self):
        # gc
        pass

    @unittest.skip("no need")
    def test_sample(self):
        #logger.info("sample test")
        print("sample test that is skipped")

    @unittest.skip("ok")
    def test_dag_creation(self):
        #print(self.simple_dag)
        for node in self.simple_dag.nodes_dag_order():
            print(node.output_name)
        print("ok")

    @unittest.skip("ok")
    def test_fit_dag(self):
        pass

    @unittest.skip("ok")
    def test_dag_forward(self):
        batch_size = 1
        train_input_names = [
            "x1",
            "x2",
            "x3",
        ]
        forward_input = torch.rand(batch_size, 1, len(train_input_names))

        self.simple_dag.forward(forward_input)

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
