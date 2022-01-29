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
from dagbo.dag_gpytorch_model import DagGPyTorchModel
from dagbo.fit_dag import fit_dag, fit_node_with_scipy, fit_node_with_adam

from dagbo.other_opt.bo_utils import get_fitted_model, inner_loop
from dagbo.utils.ax_experiment_utils import candidates_to_generator_run
from dagbo.utils.perf_model_utils import build_perf_model_from_spec, input_dict_from_ax_experiment


#class TREE_DAG(Dag, DagGPyTorchModel):
class TREE_DAG(SO_Dag, DagGPyTorchModel):
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

class CustomMetric(Metric):
    def fetch_trial_data(self, trial, **kwargs):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": params["x1"] + params["x2"] - params["x3"],
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
        #self.param_names = ["x1", "x2", "x3", "z1", "z2", "y"]

        # NOTE: this must
        self.param_names = ["x1", "x2", "x3"]

        self.search_space = SearchSpace([RangeParameter("x1", ParameterType.FLOAT, lower=-1, upper=1),
            RangeParameter("x2", ParameterType.FLOAT, lower=-1, upper=1),
            RangeParameter("x3", ParameterType.FLOAT, lower=-1, upper=1),
            ])

        # opt config
        self.optimization_config = OptimizationConfig(
            Objective(metric=CustomMetric(name="custom_obj"), minimize=False))

        # experiment
        self.exp = Experiment(name="test_exp",
                         search_space=self.search_space,
                         optimization_config=self.optimization_config,
                         runner=dummy_runner())

        # BOOTSTRAP EVALUATIONS
        num_bootstrap = 2
        sobol = Models.SOBOL(self.exp.search_space)
        generated_run = sobol.gen(num_bootstrap)
        trial = self.exp.new_batch_trial(generator_run=generated_run)
        trial.run()
        trial.mark_completed()

        self.epoch = 3

    def tearDown(self):
        pass

    def test_ax_experiment_custom_metric(self):
        # run ax-BO
        for i in range(self.epoch):
            # Reinitialize GP+EI model at each step with updated data.
            gpei = Models.BOTORCH(experiment=self.exp, data=self.exp.fetch_data())
            generator_run = gpei.gen(n=1)
            trial = self.exp.new_trial(generator_run=generator_run)
            trial.run()
            trial.mark_completed()

        print("done")
        print(self.exp.fetch_data().df)
        # to impl a ax model see: https://ax.dev/versions/0.1.3/api/modelbridge.html#model-bridges

    def test_ax_with_custom_bo(self):
        # run custom-BO
        for i in range(self.epoch):

            # get model & get candidates
            model = get_fitted_model(self.exp, self.param_names)
            candidates = inner_loop(self.exp,
                                    model,
                                    self.param_names,
                                    acq_name="qUCB",
                                    acq_func_config=self.acq_func_config)
            gen_run = candidates_to_generator_run(self.exp, candidates, self.param_names)

            # apply to system & append to dataset
            if self.acq_func_config["q"] == 1:
                trial = self.exp.new_trial(generator_run=gen_run)
            else:
                trial = self.exp.new_batch_trial(generator_run=gen_run)
            trial.run()
            trial.mark_completed()

        print("done")
        print(self.exp.fetch_data().df)

    #@unittest.skip("not ready")
    def test_ax_with_dagbo(self):
        """
        compared to standard bo-ax loop,
            dagbo needs to handle monitoring metric additionally

        dag:
        x1      x2        x3
          \     /         |
            z1           z2
              \        /
                  y
        """

        param_space = {
                "x1": "continuous",
                "x2": "continuous",
                "x3": "continuous",
                }
        metric_space = {
                "z1": "continuous",
                "z2": "continuous",
                }
        obj_space = {"y": "continuous"}

        edges = {
                "x1": ["z1"],
                "x2": ["z1"],
                "x3": ["z2"],
                "z1": ["y"],
                "z2": ["y"],
                }
        num_samples = 1024

        train_inputs_dict=input_dict_from_ax_experiment(self.exp, self.param_names)

    # make target dict
    assert (len(obj_space) == 1), "not support multi-obj for now"
    obj_name = list(obj_space.keys())[0]
    train_targets_dict[obj_name] = torch.tensor(exp_df["mean"], dtype=torch.float32)

        print(train_inputs_dict, train_targets_dict)

        for i in range(self.epoch):
            # get model & get candidates
            dag = build_perf_model_from_spec(train_inputs_dict,
                               train_targets_dict,
                               num_samples,
                               param_space,
                               metric_spacem,
                               obj_space,
                               edges,
                               )
            fit_dag(dag)
            candidates = inner_loop(exp,
                                    dag,
                                    self.param_names,
                                    acq_name="qUCB",
                                    acq_func_config=self.acq_func_config)
            gen_run = candidates_to_generator_run(exp, candidates, self.param_names)

            # run
            if self.acq_func_config["q"] == 1:
                trial = exp.new_trial(generator_run=gen_run)
            else:
                trial = exp.new_batch_trial(generator_run=gen_run)
            trial.run()
            trial.mark_completed()

            # append to dataset
            train_inputs_dict, train_targets_dict = get_tensor_to_dict(exp, train_inputs_dict, train_targets_dict)

        print("done")
        print(exp.fetch_data().df)


class test_dag_with_ax_apis(unittest.TestCase):
    """
    this just to test if dagbo can run,
        but its result is NOT comparable with others, because it uses fake data point
    """
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

        self.dag = TREE_DAG(train_input_names, train_target_names,
                                   train_inputs, train_targets, num_samples)

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

    @unittest.skip("runnable")
    def test_dag_bayes_loop(self):
        """
        the result is not meanful, just to test if it can run
        """

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

        for i in range(3):
            fit_dag(self.dag)
            candidates = inner_loop(exp,
                                    self.dag,
                                    self.param_names,
                                    acq_name="qUCB",
                                    acq_func_config=self.acq_func_config)
            gen_run = candidates_to_generator_run(exp, candidates, self.param_names)

            # run
            if self.acq_func_config["q"] == 1:
                trial = exp.new_trial(generator_run=gen_run)
            else:
                trial = exp.new_batch_trial(generator_run=gen_run)
            trial.run()
            trial.mark_completed()

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
