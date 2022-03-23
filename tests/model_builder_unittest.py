import os
import sys
import math
import warnings
import logging
import unittest
import torch
import numpy as np
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

from dagbo.interface.parse_performance_model import parse_model
from dagbo.models.model_builder import build_model, build_perf_model_from_spec_ssa, build_input_by_topological_order
from dagbo.utils.perf_model_utils import get_dag_topological_order, find_inverse_edges


class perf_model_test(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        # performance model
        param_space, metric_space, obj_space, edges = parse_model(
            "dagbo/interface/rosenbrock_3d_dummy.txt")
        #"dagbo/interface/rosenbrock_3d_correct_model.txt")

        self.param_space = param_space
        self.metric_space = metric_space
        self.obj_space = obj_space
        self.edges = edges
        #print(param_space)
        #print(edges)
        acq_func_config = {
            "q": 1,
            "num_restarts": 48,
            "raw_samples": 128,
            "num_samples": 2048,
            "y_max": torch.tensor([1.]),  # for EI
            "beta": 1,  # for UCB
        }
        self.acq_func_config = acq_func_config

        # make fake input tensor
        self.train_inputs_dict = {
            i: np.random.rand(acq_func_config["q"])
            for i in list(param_space.keys())
        }
        self.train_targets_dict = {
            i: np.random.rand(acq_func_config["q"])
            for i in list(metric_space.keys()) + list(obj_space.keys())
        }
        print(self.train_targets_dict)
        norm = True
        device = "cpu"

        # build, build_perf_model_from_spec
        #self.dag = build_perf_model_from_spec_ssa(
        #    self.train_inputs_dict, self.train_targets_dict,
        #    acq_func_config["num_samples"], param_space, metric_space,
        #    obj_space, edges, device)

    #@unittest.skip("ok")
    def test_input_build(self):
        node_order = get_dag_topological_order(self.obj_space, self.edges)
        train_input_names, train_target_names, train_inputs, train_targets = build_input_by_topological_order(
            self.train_inputs_dict, self.train_targets_dict, self.param_space,
            self.metric_space, self.obj_space, node_order)
        print("input build:")
        print()
        print("input name: ", train_input_names)
        print("target name: ", train_target_names)
        print("input: ", train_inputs.shape)
        print(train_inputs)
        print("target: ", train_targets.shape)
        print(train_targets)
        print()

class ssa_dag_test(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        performance_model_path = "dagbo/interface/rosenbrock_3d_dummy.txt"
        tuner = "dagbo-ssa"
        acq_func_config = {}
        train_inputs_dict = {}
        train_targets_dict = {}
        norm = 1
        minimize = 1
        device = "cpu"
        param_space, metric_space, obj_space, edges = parse_model(
            performance_model_path)

        self.model = build_model(tuner, train_inputs_dict, train_targets_dict,
                                 param_space, metric_space, obj_space, edges,
                                 acq_func_config, norm, minimize, device)

    @unittest.skip("ok")
    def test_print_model(self):
        print(self.model)

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


class normal_gp_test(unittest.TestCase):
    """
    test the sampling and inner loop of a standard gp model from gpytorch
        (as building block for dagbo)
    """
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
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
                print(X)
                print(mvn.loc)
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

        train_inputs = train_inputs.reshape(1, num_init, 3)
        train_targets = train_targets.reshape(1, num_init)

        self.model = gp(["x1", "x2", "x3"], "t", train_inputs, train_targets)

    @unittest.skip("print sample shape")
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

    @unittest.skip("print inner loop shape")
    def test_normal_gp_inner_loop_shape(self):
        """
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
