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

import gpytorch
from gpytorch.models.exact_gp import ExactGP
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
import botorch
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils import gpt_posterior_settings

from dagbo.dag import Dag, SO_Dag
from dagbo.models.node import Node
from dagbo.models.sample_average_posterior import SampleAveragePosterior
from dagbo.dag_gpytorch_model import DagGPyTorchModel, full_DagGPyTorchModel
from dagbo.fit_dag import fit_dag, fit_node_with_scipy, test_fit_node_with_scipy, fit_node_with_adam
from dagbo.other_opt.model_factory import fit_gpr


class normal_gp_test(unittest.TestCase):
    def setUp(self):
        class gp(Node):
            def __init__(self, input_names, output_name, train_inputs,
                         train_targets):
                super().__init__(input_names, output_name, train_inputs,
                                 train_targets)
                self.num_outputs = 1

            def posterior(self,
                          X: Tensor,
                          observation_noise=False,
                          **kwargs) -> GPyTorchPosterior:
                self.eval()  # make sure model is in eval mode
                with gpt_posterior_settings():
                    mvn = self(X)
                print("mvn:::")
                print(X.shape)
                print(mvn)
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
        num_init = 2

        # hand-craft data
        train_inputs = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.1, 0.9],
        ],
                                    dtype=torch.float32)
        func = lambda x: torch.sin(x[0] * (8 * math.pi)) + torch.cos(x[1] * (
            3 * math.pi)) + torch.log(x[2] + 0.1) + 3
        train_targets = torch.tensor([func(i) for i in train_inputs])

        # create data
        #train_inputs = torch.linspace(0, 1, 7)
        #train_inputs = torch.linspace(0, 1, num_init)
        #train_inputs = torch.rand(num_init)
        #func = lambda x: torch.sin(x * (8 * math.pi)) + torch.cos(x * (
        #    3 * math.pi)) + torch.log(x + 0.1) + 3
        ##func = lambda x: torch.sin(x * math.pi)

        ## reshape
        #train_targets = func(train_inputs).reshape(-1, 1).expand(
        #    1, num_init, 2)  # shape:[1, num_init, 3]

        ##print(train_targets[0])
        #new_val = func(train_targets[...,
        #                             -1].flatten()).reshape(1, num_init, 1)
        #train_targets = torch.cat([train_targets, new_val], dim=-1)
        ##print(new_val)
        ##print(train_targets[0])
        ##raise ValueError("ok")

        #train_inputs = train_inputs.reshape(-1, 1).expand(
        #    1, num_init, 3)  # shape:[1, num_init, 3]

        train_inputs = train_inputs.reshape(1, num_init, 3)
        train_targets = train_targets.reshape(1, num_init)
        #self.assertEqual(train_inputs.shape, torch.Size([1, num_init, 3]))
        #self.assertEqual(train_targets.shape, torch.Size([1, num_init, 3]))

        #print("training input:")
        #print(train_inputs)
        #print(train_targets)

        #train_inputs = train_inputs.reshape(num_init,3)
        #train_targets = train_targets.reshape(num_init,3)

        self.model = gp(["x1", "x2", "x3"], "t", train_inputs, train_targets)

    def test_normal_gp_sampling_shape(self):
        fit_gpr(self.model)

        #print("print param::::")
        #for i in self.model.covar.parameters():
        #    print(i)
        #for i in self.model.mean.parameters():
        #    print(i)

        train_input_names = ["x1", "x2", "x3"]
        q = 1
        #new_input = torch.rand(self.batch_size, q, len(train_input_names))
        new_input = torch.rand(1, q, len(train_input_names))

        print()
        print("normal gp sampling:::")
        print("input shape: ", new_input.shape)
        print(new_input)

        pst = self.model.posterior(new_input, **{"verbose": True})

        print()
        print("posterior:::")
        print(pst.mean)
        print(pst.variance)
        print(pst.event_shape)
        sampler = SobolQMCNormalSampler(
            num_samples=2, seed=1234)  # sampler just expand 0-dim as samples
        samples = sampler(pst)
        print()
        print("sampling from posterior:::")
        print(
            samples.shape
        )  # [sampler's num_samples, batch_size of input, q, DAG's num_of_output]
        print(samples)

    def test_normal_gp_inner_loop_shape(self):
        print()
        print("normal gp inner loop:::")
        fit_gpr(self.model)
        q = 1
        num_restarts = 24  # create batch shape for optimise acquisition func
        raw_samples = 48  # this create initial batch shape for optimise acquisition func

        sampler = SobolQMCNormalSampler(num_samples=128, seed=1234)
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
            ], dtype=torch.float32),
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=False,  # joint optimisation of q
        )
        query = candidates.detach()
        logging.info("candidates: ")
        print(query, val.detach())


class ross_dag_test(unittest.TestCase):
    def setUp(self):

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
                                 train_inputs, train_targets)

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
            ], dtype=torch.float32),
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=False,  # joint optimisation of q
        )
        query = candidates.detach()
        logging.info("candidates: ")
        print(query, val.detach())


class full_dag_test(unittest.TestCase):
    def setUp(self):
        # define dag
        class TREE_DAG(SO_Dag, full_DagGPyTorchModel):
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
                                 train_inputs, train_targets)

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

    #@unittest.skip(".")
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
        print(samples)


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
