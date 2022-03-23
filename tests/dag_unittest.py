import os
import sys
import math
import warnings
import logging
import unittest
import torch
import numpy as np
import pandas as pd
from torch import Size, Tensor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import gpytorch
from gpytorch.models.exact_gp import ExactGP
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
import botorch
from botorch.models import SingleTaskGP
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils import gpt_posterior_settings

from dagbo.models.dag.dag import Dag, SO_Dag
from dagbo.models.dag.node import SingleTaskGP_Node
from dagbo.models.dag.sample_average_posterior import SampleAveragePosterior
from dagbo.models.dag.dag_gpytorch_model import DagGPyTorchModel, direct_DagGPyTorchModel
from dagbo.models.dag.fit_dag import fit_dag, fit_node_with_scipy, test_fit_node_with_scipy, fit_node_with_adam
from dagbo.models.gp_factory import fit_gpr


class normal_gp_test(unittest.TestCase):
    """
    test the sampling and inner loop of a standard gp model from gpytorch
        (as building block for dagbo)
    """
    def setUp(self):
        np.random.seed(0), torch.manual_seed(0)

        class gp(SingleTaskGP_Node):
            def __init__(self, input_names, output_name, train_inputs,
                         train_targets):
                super().__init__(input_names, output_name, train_inputs,
                                 train_targets)

            # expose posterior to print shape
            def posterior(self,
                          X: Tensor,
                          observation_noise=False,
                          **kwargs) -> GPyTorchPosterior:
                self.eval()  # make sure model is in eval mode
                with gpt_posterior_settings():
                    mvn = self(X)
                print()
                print("X::: ", X.shape)
                print(X)
                print("mvn:::")
                print(mvn)
                print(mvn.loc)
                print()
                #print(mvn.loc)  # can verify identical mvn
                posterior = GPyTorchPosterior(mvn=mvn)
                return posterior

        # prepare input
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
        num_init = 3

        # hand-craft data
        train_inputs = np.array([
            [0.1, 0.2, 0.3],
            [0.3, 0.2, 0.7],
            [0.4, 0.1, 0.9],
        ])

        # targets
        func = lambda x: np.sin(x[0] * (8 * math.pi)) + np.cos(x[1] * (
            3 * math.pi)) + np.log(x[2] + 0.1) + 3
        train_targets = np.array([func(i) for i in train_inputs])
        train_targets = train_targets.reshape(num_init, 1)

        # format
        train_inputs = MinMaxScaler().fit_transform(train_inputs)
        train_targets = StandardScaler().fit_transform(train_targets)
        train_inputs = torch.from_numpy(train_inputs)
        train_targets = torch.from_numpy(train_targets)
        train_inputs = train_inputs.reshape(1, num_init, 3)
        train_targets = train_targets.reshape(1, num_init)

        self.model = gp(["x1", "x2", "x3"], "t", train_inputs, train_targets)

    #@unittest.skip("print sample shape")
    def test_normal_gp_sampling_shape(self):
        fit_gpr(self.model)

        train_input_names = ["x1", "x2", "x3"]
        q = 1
        #new_input = torch.rand(self.batch_size, q, len(train_input_names))
        new_input = torch.rand(1, q, len(train_input_names))

        print()
        print("normal gp sampling:::")
        print("input shape: ", new_input.shape)
        pst = self.model.posterior(new_input)

        print()
        print("posterior:::")
        print(pst.mvn)
        sampler = SobolQMCNormalSampler(
            num_samples=4, seed=0)
        samples = sampler(pst)
        print()
        print("sampling from posterior:::")
        print(
            samples.shape
        )  # [sampler's num_samples, batch_size of input, q, DAG's num_of_output]
        print(samples)

    @unittest.skip("print inner loop shape")
    def test_normal_gp_inner_loop(self):
        """
        NOTE: run this func can observe MC-gradient-descent in standard BO

        gp posterior only return one `deterministic` `posterior` object
            representing a gaussian posterior distribution
            this is ok for normal gp
            but our dagbo needs `approximate` posterior
        """
        print()
        print("normal gp inner loop:::")
        fit_gpr(self.model)
        q = 1
        num_restarts = 2  # create batch shape for optimise acquisition func
        raw_samples = 3  # this create initial batch shape for optimise acquisition func

        sampler = SobolQMCNormalSampler(num_samples=4, seed=0)
        acq = botorch.acquisition.monte_carlo.qExpectedImprovement(
            model=self.model,
            best_f=torch.tensor([1.]),
            sampler=sampler,
            objective=None,  # use when model has multiple output
        )

        # inner loop
        candidates, val = botorch.optim.optimize_acqf(
            acq_function=acq,
            bounds=torch.tensor([
                [0, 0, 0],
                [1, 1, 1],
            ], dtype=torch.float64),
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=False,  # joint optimisation of q
        )
        query = candidates.detach()
        logging.info("candidates: ")
        print(query, val.detach())


class ross_dag_dummy_perf_model_test(unittest.TestCase):
    """
    test ross' impl of sample average dagbo
    """
    def setUp(self):
        np.random.seed(0), torch.manual_seed(0)

        # define dag  direct_DagGPyTorchModel, DagGPyTorchModel
        class perf_model_DAG(SO_Dag, DagGPyTorchModel):
            def __init__(self, train_input_names: list[str],
                         train_target_names: list[str], train_inputs: Tensor,
                         train_targets: Tensor, num_samples: int):
                super().__init__(train_input_names, train_target_names,
                                 train_inputs, train_targets, "cpu")

                # required for all classes that extend SparkDag
                self.num_samples = num_samples

            def define_dag(self, batch_shape: Size = Size([])) -> None:
                x1 = self.register_input("x1")
                x2 = self.register_input("x2")
                x3 = self.register_input("x3")
                y = self.register_metric("y", [x1, x2, x3])

        # prepare input
        train_input_names = [
            "x1",
            "x2",
            "x3",
        ]
        train_target_names = [
            "y",
        ]
        num_init = 3
        num_samples = 2

        # hand-craft data
        train_inputs = np.array([
            [0.1, 0.2, 0.3],
            [0.3, 0.2, 0.7],
            [0.4, 0.1, 0.9],
        ])

        # targets
        func = lambda x: np.sin(x[0] * (8 * math.pi)) + np.cos(x[1] * (
            3 * math.pi)) + np.log(x[2] + 0.1) + 3
        train_targets = np.array([func(i) for i in train_inputs])
        train_targets = train_targets.reshape(num_init, 1)

        # format
        train_inputs = MinMaxScaler().fit_transform(train_inputs)
        train_targets = StandardScaler().fit_transform(train_targets)
        train_inputs = torch.from_numpy(train_inputs)
        train_targets = torch.from_numpy(train_targets)
        train_inputs = train_inputs.reshape(1, num_init, 3)
        train_targets = train_targets.reshape(1, num_init, 1)

        self.dag = perf_model_DAG(train_input_names, train_target_names,
                                  train_inputs, train_targets, num_samples)

    #@unittest.skip("..")
    def test_dag_posterior(self):
        print()
        print("dag sampling::::")
        fit_dag(self.dag)
        train_input_names = ["x1", "x2", "x3"]
        q = 1
        new_input = torch.rand(1, q, len(train_input_names))
        print("input shape: ", new_input.shape)

        pst = self.dag.posterior(new_input)
        print()
        print("posterior:")
        print(pst.mvn)
        sampler = SobolQMCNormalSampler(num_samples=4, seed=0)
        samples = sampler(pst)
        print()
        print("sampling from posterior")
        print(
            samples.shape
        )  # [sampler's num_samples, batch_size of input, q, DAG's num_of_output]
        print(samples)

    @unittest.skip("..")
    def test_dag_inner_loop(self):
        print()
        print("dag inner loop:::")
        fit_dag(self.dag)
        q = 1
        num_restarts = 2  # create batch shape for optimise acquisition func
        raw_samples = 3  # this create initial batch shape for optimise acquisition func

        sampler = SobolQMCNormalSampler(num_samples=4, seed=0)
        acq = botorch.acquisition.monte_carlo.qExpectedImprovement(
            model=self.dag,
            best_f=torch.tensor([1.]),
            sampler=sampler,
            objective=None,  # use when model has multiple output
        )

        # inner loop
        candidates, val = botorch.optim.optimize_acqf(
            acq_function=acq,
            bounds=torch.tensor([
                [0, 0, 0],
                [1, 1, 1],
            ], dtype=torch.float64),
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=False,  # joint optimisation of q
        )
        query = candidates.detach()
        logging.info("candidates: ")
        print(query, val.detach())

class ross_dag_test(unittest.TestCase):
    """
    test ross' impl of sample average dagbo
    """
    def setUp(self):
        np.random.seed(0), torch.manual_seed(0)

        # define dag
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
                super().__init__(train_input_names, train_target_names,
                                 train_inputs, train_targets, "cpu")

                # required for all classes that extend SparkDag
                self.num_samples = num_samples

            def define_dag(self, batch_shape: Size = Size([])) -> None:
                x_1 = self.register_input("x1")
                x_2 = self.register_input("x2")
                x_3 = self.register_input("x3")

                z_1 = self.register_metric("z1", [x_1, x_2])
                z_2 = self.register_metric("z2", [x_3])

                y = self.register_metric("y", [z_1, z_2])

        # prepare input
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

        # create data
        #train_inputs = torch.linspace(0, 1, 7)
        train_inputs = torch.linspace(0, 2, 7)
        func = lambda x: torch.sin(x * (8 * math.pi)) + torch.cos(x * (
            3 * math.pi)) + torch.log(x + 0.1) + 3
        #func = lambda x: torch.sin(x * math.pi)

        # reshape
        train_targets = func(train_inputs).reshape(-1, 1).expand(
            1, 7, 2)  # shape:[1, 7, 3]

        #print(train_targets[0])
        new_val = func(train_targets[..., -1].flatten()).reshape(1, 7, 1)
        train_targets = torch.cat([train_targets, new_val], dim=-1)
        #print(new_val)
        #print(train_targets[0])
        #raise ValueError("ok")

        train_inputs = train_inputs.reshape(-1, 1).expand(1, 7,
                                                          3)  # shape:[1, 7, 3]

        self.assertEqual(train_inputs.shape, torch.Size([1, 7, 3]))
        self.assertEqual(train_targets.shape, torch.Size([1, 7, 3]))

        self.func = func
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.domain = (0, 2)
        self.simple_dag = TREE_DAG(train_input_names, train_target_names,
                                   train_inputs, train_targets, num_samples)

    @unittest.skip(".")
    def test_model_mro(self):
        print("model MRO")
        print(TREE_DAG.__mro__)

    @unittest.skip("ok")
    def test_dag_compare_fit(self):
        """
        test fitting the dag from data, with different fit method
            each initialisation of DAG, it holds original data

        Results:
            fit_node_with_scipy tends to be more numerically stable, botorch's default fit
            fit_node_with_adam needs to choose epochs, lr, etc
        """
        # fit
        #fit_dag(self.simple_dag, fit_node_with_adam, verbose=True)
        fit_dag(self.simple_dag, test_fit_node_with_scipy, verbose=True)

        for node in self.simple_dag.nodes_dag_order():
            print("Verifying: ", node.output_name)
            node.eval()

            if node.output_name == "z2":
                with torch.no_grad():
                    test_x = torch.linspace(self.domain[0], self.domain[1],
                                            100)
                    test_y = self.func(test_x)

                    pred = node.likelihood(node(test_x))
                    mean = pred.mean
                    lower, upper = pred.confidence_region()

                    mean = mean.flatten()
                    lower, upper = lower.flatten(), upper.flatten()

                    after = mean_squared_error(test_y, mean)
        print(f"MSE after fit: {after:.2f}")

    @unittest.skip("ok")
    def test_dag_posterior(self):
        """
        test posterior returned by the DAG,
            as well as samplings from this posterior
        """
        fit_dag(self.simple_dag, fit_node_with_scipy)

        train_input_names = ["x1", "x2", "x3"]
        q = 1
        #new_input = torch.rand(self.batch_size, q, len(train_input_names))
        new_input = torch.rand(1, q, len(train_input_names))

        print("input shape: ", new_input.shape)

        pst = self.simple_dag.posterior(new_input, **{"verbose": True})

        print()
        print("posterior:")
        print(pst.mean, pst.event_shape)
        sampler = SobolQMCNormalSampler(num_samples=3, seed=1234)
        samples = sampler(pst)
        print()
        print("sampling from posterior")
        print(
            samples.shape
        )  # [sampler's num_samples, batch_size of input, q, DAG's num_of_output]
        print(samples)

    @unittest.skip("..")
    def test_dag_inner_loop(self):
        """
        test optimise the acquisition function
        """
        fit_dag(self.simple_dag, fit_node_with_scipy)
        q = 1
        num_restarts = 24  # create batch shape for optimise acquisition func
        raw_samples = 48  # this create initial batch shape for optimise acquisition func

        # --- Botorch's acquisition function input to posterior
        print()
        logging.info("Botorch input shape: ")
        sampler = SobolQMCNormalSampler(num_samples=2048, seed=1234)
        acq = botorch.acquisition.monte_carlo.qExpectedImprovement(
            model=self.simple_dag,
            best_f=torch.tensor([1.]),
            sampler=sampler,
            objective=None,  # use when model has multiple output
        )

        # inner loop
        candidates, val = botorch.optim.optimize_acqf(
            acq_function=acq,
            bounds=torch.tensor([
                [0, 0, 0],
                [1, 1, 1],
            ], dtype=torch.float64),
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=False,  # joint optimisation of q
        )
        query = candidates.detach()
        logging.info("candidates: ")
        print(query, val.detach())


class direct_dag_test(unittest.TestCase):
    """
    test my impl of sampling from dagbo's approx. posterior
    """
    def setUp(self):
        np.random.seed(0), torch.manual_seed(0)

        # define dag
        class TREE_DAG(SO_Dag, direct_DagGPyTorchModel):
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
                super().__init__(train_input_names, train_target_names,
                                 train_inputs, train_targets, "cpu")

                # required for all classes that extend SparkDag
                self.num_samples = num_samples

            def define_dag(self, batch_shape: Size = Size([])) -> None:
                x_1 = self.register_input("x1")
                x_2 = self.register_input("x2")
                x_3 = self.register_input("x3")

                z_1 = self.register_metric("z1", [x_1, x_2])
                z_2 = self.register_metric("z2", [x_3])

                y = self.register_metric("y", [z_1, z_2])

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
        num_samples = 2

        # create data
        #train_inputs = torch.linspace(0, 1, 7)
        train_inputs = torch.linspace(0, 2, 7)
        func = lambda x: torch.sin(x * (8 * math.pi)) + torch.cos(x * (
            3 * math.pi)) + torch.log(x + 0.1) + 3
        #func = lambda x: torch.sin(x * math.pi)

        # reshape
        train_targets = func(train_inputs).reshape(-1, 1).expand(
            1, 7, 2)  # shape:[1, 7, 3]

        #print(train_targets[0])
        new_val = func(train_targets[..., -1].flatten()).reshape(1, 7, 1)
        train_targets = torch.cat([train_targets, new_val], dim=-1)
        #print(new_val)
        #print(train_targets[0])
        #raise ValueError("ok")

        train_inputs = train_inputs.reshape(-1, 1).expand(1, 7,
                                                          3)  # shape:[1, 7, 3]

        self.assertEqual(train_inputs.shape, torch.Size([1, 7, 3]))
        self.assertEqual(train_targets.shape, torch.Size([1, 7, 3]))

        self.func = func
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.domain = (0, 2)
        self.simple_dag = TREE_DAG(train_input_names, train_target_names,
                                   train_inputs, train_targets, num_samples)

    @unittest.skip(".")
    def test_dag_posterior(self):
        """
        test posterior returned by the DAG,
            as well as samplings from this posterior
        """
        fit_dag(self.simple_dag, fit_node_with_scipy)

        train_input_names = ["x1", "x2", "x3"]
        q = 1
        #new_input = torch.rand(self.batch_size, q, len(train_input_names))
        new_input = torch.rand(1, q, len(train_input_names))

        print("input shape: ", new_input.shape)

        pst = self.simple_dag.posterior(new_input, **{"verbose": True})

        print()
        print("full posterior:")
        print(pst.mean, pst.variance, pst.event_shape)
        sampler = SobolQMCNormalSampler(num_samples=3, seed=1234)
        samples = sampler(pst)
        print()
        print("sampling from posterior")
        print(
            samples.shape
        )  # [sampler's num_samples, batch_size of input, q, DAG's num_of_output]
        #print(samples)

    @unittest.skip(".")
    def test_dag_inner_loop(self):
        """
        batch average MC grad!!!

            original input X: [b, q, dim]
            original output y: [b, q]  # becuase the output dim is 1, it is squeezed
                (where b dim is used as `raw_sample`  or `num_restarts` in innner loop)

        now I want to repeat original process many times and take the average
            (sampling from approx. posterior)

        I unsqueeze a new sampling dimension: X [n, b, q, dim]
            which gives a `batch` output y: [n, b, q]

        and then i take average y -> y' = [b, q]
            gpytorch's MultivariateNormal is treated specially
        """
        print()
        print("full dagbo inner loop:::")
        fit_dag(self.simple_dag, fit_node_with_scipy)
        q = 1
        num_restarts = 3  # create batch shape for optimise acquisition func
        raw_samples = 4  # this create initial batch shape for optimise acquisition func

        # --- Botorch's acquisition function input to posterior
        print()
        logging.info("Botorch input shape: ")
        sampler = SobolQMCNormalSampler(num_samples=8, seed=1234)
        acq = botorch.acquisition.monte_carlo.qExpectedImprovement(
            model=self.simple_dag,
            best_f=torch.tensor([1.]),
            sampler=sampler,
            objective=None,  # use when model has multiple output
        )

        # inner loop
        candidates, val = botorch.optim.optimize_acqf(
            acq_function=acq,
            bounds=torch.tensor([
                [0, 0, 0],
                [1, 1, 1],
            ], dtype=torch.float64),
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=False,  # joint optimisation of q
        )
        query = candidates.detach()
        logging.info("candidates: ")
        print(query, val.detach())


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
