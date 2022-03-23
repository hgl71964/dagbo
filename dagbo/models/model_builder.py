from typing import Union
from copy import deepcopy

import torch
import gpytorch
import numpy as np
from torch import Tensor
from ax import Experiment
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from botorch.models import SingleTaskGP
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dagbo.models.dag.dag import Dag, lazy_SO_Dag
from dagbo.models.dag.dag_gpytorch_model import DagGPyTorchModel, direct_DagGPyTorchModel
from dagbo.models.dag.fit_dag import fit_dag
from dagbo.models.gp_factory import make_gps, make_node, make_SingleTaskGP_node, fit_gpr
from dagbo.models.dag.parametric_mean import LinearMean
from dagbo.utils.perf_model_utils import get_dag_topological_order, find_inverse_edges


def build_model(tuner: str, train_inputs_dict: dict, train_targets_dict: dict,
                param_space: dict, metric_space: dict, obj_space: dict,
                edges: dict, acq_func_config: dict, standardisation: bool,
                minimize: bool, device) -> Union[Dag, SingleTaskGP]:

    # format data
    ## deepcopy
    train_inputs_dict_ = deepcopy(train_inputs_dict)
    train_targets_dict_ = deepcopy(train_targets_dict)
    ## flip obj sign
    train_targets_dict_ = flip_obj(train_targets_dict_, obj_space, minimize)
    ## side-effect
    update_acq_func_config(acq_func_config, obj_space, train_targets_dict_)

    # build model
    model = None
    if tuner == "dagbo-ssa":
        model = build_perf_model_from_spec_ssa(train_inputs_dict_,
                                               train_targets_dict_,
                                               acq_func_config["num_samples"],
                                               param_space, metric_space,
                                               obj_space, edges,
                                               standardisation, device)
        fit_dag(model)
    elif tuner == "dagbo-direct":
        model = build_perf_model_from_spec_direct(
            train_inputs_dict_, train_targets_dict_,
            acq_func_config["num_samples"], param_space, metric_space,
            obj_space, edges, standardisation, device)
        fit_dag(model)
    elif tuner == "bo":
        model = build_gp_from_spec(train_inputs_dict_, train_targets_dict_,
                                   param_space, metric_space, obj_space, edges,
                                   standardisation)
        fit_gpr(model)
    else:
        raise ValueError("unable to recognize tuner")

    return model


def build_gp_from_spec(
    train_inputs_dict: dict[str, np.ndarray],
    train_targets_dict: dict[str, np.ndarray],
    param_space: dict[str, str],
    metric_space: dict[str, str],
    obj_space: dict[str, str],
    edges: dict[str, list[str]],
    standardisation: bool,
) -> SingleTaskGP:

    # build input
    node_order = get_dag_topological_order(obj_space, edges)

    ##
    train_input_names, train_target_names, train_inputs, train_targets = build_input_by_topological_order(
        train_inputs_dict, train_targets_dict, param_space, metric_space,
        obj_space, node_order, standardisation)

    ##
    assert train_inputs.shape[0] == 1
    assert train_targets.shape[0] == 1
    # NOTE: don't need gpu for inference?
    x = train_inputs.squeeze(0)
    y = train_targets.squeeze(0)[..., -1]
    y = y.reshape(-1, 1)  # [q, 1] for 1 dim output
    #print()
    #print("shape")
    #print(x.shape)  # [q, dim]
    #print(y.shape)  # [q, 1]
    #print(y)
    #print()
    # NOTE: the following gives the same results with a fixed seed, therefore gp node is ok
    #gpr = make_gps(x=x, y=y, gp_name="MA")
    #gpr = make_node(x=x, y=y, gp_name="MA")
    gpr = make_SingleTaskGP_node(x=x, y=y, gp_name="MA")
    return gpr


def build_perf_model_from_spec_ssa(
    train_inputs_dict: dict[str, np.ndarray],
    train_targets_dict: dict[str, np.ndarray],
    num_samples: int,
    param_space: dict[str, str],
    metric_space: dict[str, str],
    obj_space: dict[str, str],
    edges: dict[str, list[str]],
    standardisation: bool,
    device,
) -> Dag:
    """
    build perf_dag from given spec (use sample average posterior)

    Core Args:
        param_space: key: param name - val: `categorical` or `continuous`
        metric_space: key: metric name - val: property of the node
        obj_space: key: obj name - val: property of the node
        edges: key: node names - val: list of name this node point to
            NOTE: edges are forward directions, i.e. from param_space -> metric_space -> obj_space
    """
    class perf_DAG(lazy_SO_Dag, DagGPyTorchModel):
        """dynamically define dag
        """
        def __init__(self, train_input_names: list[str],
                     train_target_names: list[str], train_inputs: Tensor,
                     train_targets: Tensor, num_samples: int, device):
            super().__init__(train_input_names, train_target_names,
                             train_inputs, train_targets, device)
            self.num_samples = num_samples

    # build
    reversed_edge = find_inverse_edges(edges)
    node_order = get_dag_topological_order(obj_space, edges)

    ## build input
    train_input_names, train_target_names, train_inputs, train_targets = build_input_by_topological_order(
        train_inputs_dict, train_targets_dict, param_space, metric_space,
        obj_space, node_order, standardisation)
    ## put into device, node will be taken into device automatically
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    #print(train_input_names, train_target_names)
    #print(train_inputs.is_cuda, train_targets.is_cuda)
    #print(train_inputs.shape, train_targets.shape)

    ## build model
    dag = perf_DAG(train_input_names, train_target_names, train_inputs,
                   train_targets, num_samples, device)

    # register nodes, MUST register in topological order
    # input space, TODO address for categorical variable in the future
    for node in node_order:
        if node in param_space:
            dag.register_input(node)
        elif node in metric_space or node in obj_space:
            children = [i for i in reversed_edge[node]]
            mean = build_mean(node, metric_space, obj_space, children)
            covar = build_covar(node, metric_space, obj_space, children)
            dag.register_metric(node, children, mean=mean, covar=covar)
        else:
            raise RuntimeError("unknown node")
    dag.to(device)
    return dag


def build_perf_model_from_spec_direct(train_inputs_dict: dict[str, np.ndarray],
                                      train_targets_dict: dict[str,
                                                               np.ndarray],
                                      num_samples: int, param_space: dict[str,
                                                                          str],
                                      metric_space: dict[str, str],
                                      obj_space: dict[str, str],
                                      edges: dict[str, list[str]],
                                      standardisation: bool, device) -> Dag:
    """
    use approx. posterior
    """
    class perf_DAG(lazy_SO_Dag, direct_DagGPyTorchModel):
        """dynamically define dag
        """
        def __init__(self, train_input_names: list[str],
                     train_target_names: list[str], train_inputs: Tensor,
                     train_targets: Tensor, num_samples: int, device):
            super().__init__(train_input_names, train_target_names,
                             train_inputs, train_targets, device)
            self.num_samples = num_samples

    # build
    reversed_edge = find_inverse_edges(edges)
    node_order = get_dag_topological_order(obj_space, edges)

    ##
    train_input_names, train_target_names, train_inputs, train_targets = build_input_by_topological_order(
        train_inputs_dict, train_targets_dict, param_space, metric_space,
        obj_space, node_order, standardisation)
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    dag = perf_DAG(train_input_names, train_target_names, train_inputs,
                   train_targets, num_samples, device)

    # register nodes, MUST register in topological order
    # input space, TODO address for categorical variable in the future
    for node in node_order:
        if node in param_space:
            dag.register_input(node)
        elif node in metric_space or node in obj_space:
            children = [i for i in reversed_edge[node]]
            mean = build_mean(node, metric_space, obj_space, children)
            covar = build_covar(node, metric_space, obj_space, children)
            dag.register_metric(node, children, mean=mean, covar=covar)
        else:
            raise RuntimeError("unknown node")
    dag.to(device)
    return dag


def build_input_by_topological_order(
        train_inputs_dict: dict[str, np.ndarray],
        train_targets_dict: dict[str, np.ndarray],
        param_space: dict,
        metric_space: dict,
        obj_space: dict,
        node_order: list[str],
        standardisation: bool,
        dtype=torch.float64) -> tuple[list[str], list[str], Tensor, Tensor]:
    """
    node names and their corresponding tensor must match

    Assume:
        Tensor from dict is 1-dim, each element represents a sample
            and the len of Tensor must be equal
    """
    train_input_names, train_target_names = [], []
    train_inputs, train_targets = [], []

    # ensure sorted order - NOTE: in fact, this doesn't matter, but to keep consistency
    for node in sorted(list(param_space.keys())):
        train_input_names.append(node)
        #train_inputs.append(torch.tensor(train_inputs_dict[node], dtype=dtype))
        train_inputs.append(train_inputs_dict[node])

    # NOTE: if train_targets_dict has extra metrics that doesn't exist in performance model, they are not considerd
    for node in node_order:
        if node in param_space:
            continue
        elif node in metric_space or node in obj_space:
            train_target_names.append(node)
            #train_targets.append(torch.tensor(train_targets_dict[node], dtype=dtype))
            train_targets.append(train_targets_dict[node])
        else:
            raise RuntimeError(
                "node not in param_space or metric_space or obj_space")

    # format tensor, must be consistent with Dag's signature shape
    # NOTE: after the transpose, [q, dim]
    train_inputs = np.stack(train_inputs).T
    train_targets = np.stack(train_targets).T

    # standardisation
    # NOTE: across q-dim!!
    if standardisation:
        # StandardScaler, MinMaxScaler
        train_inputs = MinMaxScaler().fit_transform(train_inputs)
        train_targets = MinMaxScaler().fit_transform(train_targets)

    # reshape & tensor
    in_dim = len(train_input_names)
    target_dim = len(train_target_names)
    train_inputs = train_inputs.reshape(1, -1, in_dim)
    train_targets = train_targets.reshape(1, -1, target_dim)
    train_inputs = torch.from_numpy(train_inputs).to(dtype)
    train_targets = torch.from_numpy(train_targets).to(dtype)
    return train_input_names, train_target_names, train_inputs, train_targets


def flip_obj(input_dict: dict, obj_space: dict, minimize: bool):
    """
    In the case of minimization, need to flip the obj
        (by default: bo maximizes obj)
    """
    if minimize:
        obj = list(obj_space.keys())[0]
        tmp = input_dict[obj]
        input_dict[obj] = -tmp
    return input_dict


def update_acq_func_config(acq_func_config,
                           obj_space,
                           train_targets_dict,
                           dtype=torch.float64):

    # update acq_func_config, e.g. update the best obs for EI or beta for UCB
    keys = list(obj_space.keys())
    assert len(keys) == 1
    obj = keys[0]
    tmp = train_targets_dict[obj]
    acq_func_config["y_max"] = torch.tensor(tmp.max(), dtype=dtype)


def build_mean(node: str, metric_space: dict, obj_space: dict,
               children: list[str]):
    """
    apply custom policy to build mean func
    """
    if node in metric_space:
        ppt = metric_space[node]
    elif node in obj_space:
        ppt = obj_space[node]

    mean = None
    #if node == "taskTime":
    #    n = len(children)
    #    mean = LinearMean(input_size=n)
    return mean


def build_covar(node: str, metric_space: dict, obj_space: dict,
                children: list[str]):
    """
    apply custom policy to build kernel
    """
    if node in metric_space:
        ppt = metric_space[node]
    elif node in obj_space:
        ppt = obj_space[node]

    covar = None
    if covar is None:
        print(f"building {node}")
        covar = ScaleKernel(MaternKernel(nu=2.5, lengthscale_prior=GammaPrior(3.0, 6.0)), outputscale_prior=GammaPrior( 2.0, 0.15))

    #if node == "unified_mem":
    #    print(f"building {node} with custom kernel")
    #    n = len(children)
    #    base_kernel = ScaleKernel(MaternKernel(nu=2.5,
    #                                           lengthscale_prior=GammaPrior(
    #                                               3.0, 6.0)),
    #                              outputscale_prior=GammaPrior(2.0, 0.15))
    #    covar = gpytorch.kernels.ProductStructureKernel(
    #        base_kernel=base_kernel, num_dims=n)

    #elif node == "duration" or node == "throughput":
    #    print(f"building {node} with custom kernel")
    #    print(f"with children", children)

    #    m = {}
    #    n = len(children)
    #    for i, child in enumerate(children):
    #        m[child] = i

    #    task_dim = (m["taskTime"], )
    #    rest_dim = tuple([v for k, v in m.items() if k != "taskTime"])
    #    print("task dim:", task_dim)
    #    print("rest dim:", rest_dim)

    #    task_kernel = ScaleKernel(MaternKernel(nu=2.5,
    #                                           active_dims=task_dim,
    #                                           lengthscale_prior=GammaPrior(
    #                                               3.0, 6.0)),
    #                              outputscale_prior=GammaPrior(2.0, 0.15))
    #    rest_kernel = ScaleKernel(MaternKernel(nu=2.5,
    #                                           active_dims=rest_dim,
    #                                           lengthscale_prior=GammaPrior(
    #                                               3.0, 6.0)),
    #                              outputscale_prior=GammaPrior(2.0, 0.15))
    #    covar = task_kernel + rest_kernel

    #elif node == "executorRunTime":
    #    print(f"building {node} with custom kernel")

    #    m = {}
    #    n = len(children)
    #    for i, child in enumerate(children):
    #        m[child] = i

    #    mem_dim = (m["executor.memory"], )
    #    mem_fra_dim = (m["memory.fraction"], )
    #    #rest_dim = tuple([ v for k, v in m.items() if k != "executor.memory" and k!="memory.fraction"])
    #    all_dim = tuple([i for i in range(n)])

    #    mem_kernel = ScaleKernel(MaternKernel(nu=2.5,
    #                                      active_dims=mem_dim,
    #                                      lengthscale_prior=GammaPrior(
    #                                          3.0, 6.0)),
    #                         outputscale_prior=GammaPrior(2.0, 0.15))
    #    mem_fra_kernel = ScaleKernel(MaternKernel(nu=2.5,
    #                                      active_dims=mem_fra_dim,
    #                                      lengthscale_prior=GammaPrior(
    #                                          3.0, 6.0)),
    #                         outputscale_prior=GammaPrior(2.0, 0.15))
    #    joint_mem_kernel = mem_kernel * mem_fra_kernel

    #    rest_kernel = ScaleKernel(MaternKernel(nu=2.5,
    #                                      active_dims=all_dim,
    #                                      lengthscale_prior=GammaPrior(
    #                                          3.0, 6.0)),
    #                         outputscale_prior=GammaPrior(2.0, 0.15))
    #    covar = joint_mem_kernel + rest_kernel

    #elif node == "executorRunTime":
    #    print(f"building {node} with custom kernel")
    #    n = len(children)
    #    child_set = set([
    #        "executor.memory", "memory.fraction", "executor.cores",
    #        "executor.num[*]", "default.parallelism"
    #    ])
    #    assert set(children) == child_set, f"{node} children error"

    #    m = {}
    #    for i, child in enumerate(children):
    #        m[child] = i

    #    # active dim TODO separate mem dim
    #    mem_dim = (m["executor.memory"], )
    #    mem_2d_dim = (m["executor.memory"], m["memory.fraction"])
    #    rest_dim = ([
    #        v for k, v in m.items()
    #        if k != "memory.fraction" and k != "executor.memory"
    #    ])

    #    # base kernels
    #    mem_1 = ScaleKernel(MaternKernel(nu=2.5,
    #                                     active_dims=mem_dim,
    #                                     lengthscale_prior=GammaPrior(
    #                                         3.0, 6.0)),
    #                        outputscale_prior=GammaPrior(2.0, 0.15))
    #    mem_2 = ScaleKernel(MaternKernel(nu=2.5,
    #                                     active_dims=mem_2d_dim,
    #                                     lengthscale_prior=GammaPrior(
    #                                         3.0, 6.0)),
    #                        outputscale_prior=GammaPrior(2.0, 0.15))
    #    base_1 = ScaleKernel(
    #        MaternKernel(
    #            nu=2.5,
    #            #active_dims=tuple([i for i in range(n)]),
    #            active_dims=rest_dim,
    #            lengthscale_prior=GammaPrior(3.0, 6.0)),
    #        outputscale_prior=GammaPrior(2.0, 0.15))
    #    covar = mem_1 + mem_2 + base_1

    #elif node == "duration" or node == "throughput":
    #    print(f"building {node} with custom kernel")
    #    n = len(children)
    #    child_set = set(["executor.num[*]", "default.parallelism", "taskTime"])
    #    assert set(children) == child_set, f"{node} children error"

    #    m = {}
    #    for i, child in enumerate(children):
    #        m[child] = i

    #    # custom additive kernels TODO del taskTime dim
    #    active_dims_1 = (m["executor.num[*]"], )
    #    active_dims_2 = (m["default.parallelism"], )
    #    active_dims_3 = (m["executor.num[*]"], m["default.parallelism"])
    #    active_dims_4 = (m["taskTime"], )
    #    all_dim = tuple([i for i in range(n)])
    #    base_1 = ScaleKernel(MaternKernel(nu=2.5,
    #                                      active_dims=active_dims_1,
    #                                      lengthscale_prior=GammaPrior(
    #                                          3.0, 6.0)),
    #                         outputscale_prior=GammaPrior(2.0, 0.15))
    #    base_2 = ScaleKernel(MaternKernel(nu=2.5,
    #                                      active_dims=active_dims_2,
    #                                      lengthscale_prior=GammaPrior(
    #                                          3.0, 6.0)),
    #                         outputscale_prior=GammaPrior(2.0, 0.15))
    #    base_3 = ScaleKernel(MaternKernel(nu=2.5,
    #                                      active_dims=active_dims_3,
    #                                      lengthscale_prior=GammaPrior(
    #                                          3.0, 6.0)),
    #                         outputscale_prior=GammaPrior(2.0, 0.15))
    #    base_4 = ScaleKernel(MaternKernel(nu=2.5,
    #                                      active_dims=active_dims_4,
    #                                      lengthscale_prior=GammaPrior(
    #                                          3.0, 6.0)),
    #                         outputscale_prior=GammaPrior(2.0, 0.15))
    #    base_5 = ScaleKernel(MaternKernel(nu=2.5,
    #                                      active_dims=all_dim,
    #                                      lengthscale_prior=GammaPrior(
    #                                          3.0, 6.0)),
    #                         outputscale_prior=GammaPrior(2.0, 0.15))
    #    covar = base_1 + base_2 + base_3 + base_4 + base_5
    return covar
