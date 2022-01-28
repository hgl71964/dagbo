import os
import sys
import math
import warnings
import logging
import unittest
import torch
import pandas as pd
from torch import Size, Tensor
from sklearn.metrics import mean_squared_error
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.samplers import SobolQMCNormalSampler

# ax-platform
import ax
from ax import RangeParameter, ChoiceParameter, ParameterType, \
    SearchSpace, Experiment, OptimizationConfig, Objective, Metric
from ax.modelbridge.registry import Models

from dagbo.dag import Dag, SO_Dag
from dagbo.models.sample_average_posterior import SampleAveragePosterior
from dagbo.dag_gpytorch_model import DagGPyTorchModel
from dagbo.fit_dag import fit_dag, fit_node_with_scipy, fit_node_with_adam

from dagbo.other_opt.bo_utils import get_fitted_model, inner_loop
from dagbo.utils.ax_experiment_utils import candidates_to_generator_run


#class TREE_DAG(Dag, DagGPyTorchModel):
class TREE_DAG(SO_Dag, DagGPyTorchModel):
    """
    creation a simple tree-like DAG

    x1      x2        x3
      \     /         |
        z1           z2
          \        /
              y
    """
    def __init__(self, train_input_names: list[str],
                 train_target_names: list[str], train_inputs: Tensor,
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


def init_and_fit(train_input_names: list[str], train_target_names: list[str],
                 train_inputs: Tensor, train_targets: Tensor,
                 num_samples: int) -> TREE_DAG:
    """instantiate tree_dag from data and fit
    TODO
    """
    return None


def model_to_generator_run():
    """
    model's suggestion -> Arm -> Generator run
    TODO
    """
    return


class CustomMetric(Metric):
    def fetch_trial_data(self, trial, **kwargs):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": params["x1"] + params["x2"],
                "sem": 0,  # 0 for noiseless experiment
                "trial_index": trial.index,
            })
        return ax.core.data.Data(df=pd.DataFrame.from_records(records))


class dummy_runner(ax.Runner):
    """control how experiment is deployed, i.e. locally or dispatch to external system
        mapping from arms to other APIs
    """
    def run(self, trial):
        trial_metadata = {"from_runner - name": str(trial.index)}
        return trial_metadata


class test_basic_ax_apis(unittest.TestCase):
    def setUp(self):
        #
        """--- bo ---"""
        self.acq_func_config = {
            "q": 2,
            "num_restarts": 48,
            "raw_samples": 128,
            "num_samples": 2048,
            "y_max": torch.tensor([1.]),  # for EI
            "beta": 1,
        }

        #
        """--- set up Ax ---"""
        # parameter space
        self.param_names = ["x1", "x2", "x3", "z1", "z2", "y"]
        x1 = RangeParameter("x1", ParameterType.FLOAT, lower=-1, upper=1)
        x2 = RangeParameter("x2", ParameterType.FLOAT, lower=-1, upper=1)
        x3 = RangeParameter("x3", ParameterType.FLOAT, lower=-1, upper=1)
        z1 = RangeParameter("z1", ParameterType.FLOAT, lower=-1, upper=1)
        z2 = RangeParameter("z2", ParameterType.FLOAT, lower=-1, upper=1)
        y = RangeParameter("y", ParameterType.FLOAT, lower=-1, upper=1)
        parameters = [x1, x2, x3, z1, z2, y]
        self.search_space = SearchSpace(parameters)

        # opt config
        self.optimization_config = OptimizationConfig(
            Objective(metric=CustomMetric(name="custom_obj"), minimize=False))

    def tearDown(self):
        # gc
        pass

    #@unittest.skip("ok")
    def test_ax_experiment_custom_metric(self):
        print("test ax custom metric")
        # experiment
        exp = Experiment(name="test_exp",
                         search_space=self.search_space,
                         optimization_config=self.optimization_config,
                         runner=dummy_runner())

        # BOOTSTRAP EVALUATIONS
        num_bootstrap = 2
        sobol = Models.SOBOL(exp.search_space)
        generated_run = sobol.gen(num_bootstrap)
        print("gen")
        print(generated_run)
        trial = exp.new_batch_trial(generator_run=generated_run)
        trial.run()
        trial.mark_completed()

        # run ax-BO
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
        # to impl a ax model see: https://ax.dev/versions/0.1.3/api/modelbridge.html#model-bridges

    def test_ax_with_custom_bo(self):
        # experiment
        exp = Experiment(name="test_exp",
                         search_space=self.search_space,
                         optimization_config=self.optimization_config,
                         runner=dummy_runner())

        # BOOTSTRAP EVALUATIONS
        num_bootstrap = 2
        sobol = Models.SOBOL(exp.search_space)
        generated_run = sobol.gen(num_bootstrap)
        print("gen")
        print(generated_run)
        trial = exp.new_batch_trial(generator_run=generated_run)
        trial.run()
        trial.mark_completed()

        # run custom-BO
        epochs = 3
        for i in range(epochs):
            model = get_fitted_model(exp, self.param_names)
            candidates = inner_loop(exp,
                                    model,
                                    self.param_names,
                                    acq_name="qUCB",
                                    acq_func_config=self.acq_func_config)
            gen_run = candidates_to_generator_run(exp, candidates, self.param_names)
            """ax APIs"""
            if self.acq_func_config["q"] == 1:
                trial = exp.new_trial(generator_run=gen_run)
            else:
                trial = exp.new_batch_trial(generator_run=gen_run)
            trial.run()
            trial.mark_completed()

        print("done")
        print(exp.fetch_data().df)



class test_dag_with_ax_apis(unittest.TestCase):
    def setUp(self):
        #
        """--- bo ---"""
        self.acq_func_config = {
            "q": 2,
            "num_restarts": 48,
            "raw_samples": 128,
            "num_samples": 2048,
            "y_max": torch.tensor([1.]),  # for EI
            "beta": 1,
        }

        #
        """--- dag ---"""
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
        self.param_names = ["x1", "x2", "x3", "z1", "z2", "y"]
        batch_size = 1
        num_samples = 1024

        # create data
        train_inputs = torch.linspace(0, 2, 7)
        func = lambda x: torch.sin(x * (8 * math.pi)) + torch.cos(x * (
            3 * math.pi)) + torch.log(x + 0.1) + 3
        #func = lambda x: torch.sin(x * math.pi)

        # reshape
        train_targets = func(train_inputs).reshape(-1, 1).expand(
            1, 7, 2)
        new_val = func(train_targets[..., -1].flatten()).reshape(1, 7, 1)
        train_targets = torch.cat([train_targets, new_val], dim=-1) # shape:[1, 7, 3]

        train_inputs = train_inputs.reshape(-1, 1).expand(1, 7,
                                                          3)  # shape:[1, 7, 3]
        self.func = func
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.domain = (0, 2)

        #
        """--- set up Ax ---"""
        # parameter space
        x1 = RangeParameter("x1", ParameterType.FLOAT, lower=-1, upper=1)
        x2 = RangeParameter("x2", ParameterType.FLOAT, lower=-1, upper=1)
        x3 = RangeParameter("x3", ParameterType.FLOAT, lower=-1, upper=1)
        z1 = RangeParameter("z1", ParameterType.FLOAT, lower=-1, upper=1)
        z2 = RangeParameter("z2", ParameterType.FLOAT, lower=-1, upper=1)
        y = RangeParameter("y", ParameterType.FLOAT, lower=-1, upper=1)
        parameters = [x1, x2, x3, z1, z2, y]
        self.search_space = SearchSpace(parameters)

        # opt config
        self.optimization_config = OptimizationConfig(
            Objective(metric=CustomMetric(name="custom_obj"), minimize=False))

    def test_dag_bayes_loop(self):
        print("test tree dag bayes loop")

        # experiment
        exp = Experiment(name="test_exp",
                         search_space=self.search_space,
                         optimization_config=self.optimization_config,
                         runner=dummy_runner())
        # BOOTSTRAP
        num_bootstrap = 2
        sobol = Models.SOBOL(exp.search_space)
        generated_run = sobol.gen(num_bootstrap)
        trial = exp.new_batch_trial(generator_run=generated_run)
        trial.run()
        trial.mark_completed()

        print("start bo:")
        epochs = 3
        for i in range(epochs):
            # TODO
            pass
        print("done")
        print(exp.fetch_data().df)

if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()

    # create formatter and add it to the handler
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handler to the logger
    logger.addHandler(handler)

    unittest.main()
