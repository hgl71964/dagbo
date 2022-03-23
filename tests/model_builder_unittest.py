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
        np.random.seed(0), torch.manual_seed(0)
        # performance model
        param_space, metric_space, obj_space, edges = parse_model(
            "dagbo/interface/rosenbrock_20d_dummy.txt")
        #"dagbo/interface/rosenbrock_3d_dummy.txt")
        #"dagbo/interface/rosenbrock_3d_correct_model.txt")

        self.param_space = param_space
        self.metric_space = metric_space
        self.obj_space = obj_space
        self.edges = edges
        q = 2
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
            i: np.random.rand(q)
            for i in list(param_space.keys())
        }
        self.train_targets_dict = {
            i: np.random.rand(q)
            for i in list(metric_space.keys()) + list(obj_space.keys())
        }
        #print(self.train_inputs_dict)
        #print(self.train_targets_dict)
        norm = True
        device = "cpu"

        # build, build_perf_model_from_spec
        self.dag = build_perf_model_from_spec_ssa(
            self.train_inputs_dict, self.train_targets_dict,
            acq_func_config["num_samples"], param_space, metric_space,
            obj_space, edges, device)

    @unittest.skip("ok")
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

    @unittest.skip("ok")
    def test_print_dag(self):
        print()
        print("print dag")
        print(self.dag)
        print()
        print("input map: ")
        print(self.dag.train_inputs_name2tensor_mapping)
        print("target map: ")
        print(self.dag.train_targets_name2tensor_mapping)
        print()

    @unittest.skip("ok")
    def test_print_node(self):
        print()
        print("print node")
        print(self.dag._modules["final"].input_names)
        print(self.dag._modules["final"].train_inputs)
        print(self.dag._modules["final"].train_targets)
        print()


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
