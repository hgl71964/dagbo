from typing import Union
from copy import deepcopy

import torch
import numpy as np
from torch import Tensor
from ax import Experiment
from botorch.models import SingleTaskGP
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dagbo.dag import Dag, lazy_SO_Dag
from dagbo.dag_gpytorch_model import DagGPyTorchModel, direct_DagGPyTorchModel
from dagbo.fit_dag import fit_dag
from dagbo.models.gp_factory import make_gps, fit_gpr
from dagbo.utils.perf_model_utils import get_dag_topological_order, find_inverse_edges


def build_model(tuner: str, exp: Experiment, train_inputs_dict: dict,
                train_targets_dict: dict, param_space: dict,
                metric_space: dict, obj_space: dict, edges: dict,
                acq_func_config: dict, standardisation: bool, minimize: bool,
                device) -> Union[Dag, SingleTaskGP]:

    # format data
    ## standardisation - deepcopy
    train_inputs_dict_ = standard_dict(train_inputs_dict, standardisation)
    train_targets_dict_ = standard_dict(train_targets_dict, standardisation)
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
                                               obj_space, edges, device)
        fit_dag(model)
    elif tuner == "dagbo-direct":
        model = build_perf_model_from_spec_direct(
            train_inputs_dict_, train_targets_dict_,
            acq_func_config["num_samples"], param_space, metric_space,
            obj_space, edges, device)
        fit_dag(model)
    elif tuner == "bo":
        model = build_gp_from_spec(train_inputs_dict_, train_targets_dict_,
                                   param_space, metric_space, obj_space, edges)
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
) -> SingleTaskGP:

    # build input
    node_order = get_dag_topological_order(obj_space, edges)

    ##
    train_input_names, train_target_names, train_inputs, train_targets = build_input_by_topological_order(
        train_inputs_dict, train_targets_dict, param_space, metric_space,
        obj_space, node_order)

    ##
    assert train_inputs.shape[0] == 1
    assert train_targets.shape[0] == 1
    # NOTE: don't need gpu for inference?
    x = train_inputs.squeeze(0)
    y = train_targets.squeeze(0)[..., -1]
    y = y.reshape(-1, 1)  # [q, 1] for 1 dim output

    #print()
    #print("shape")
    #print(x.shape)
    #print(y.shape)
    #print(y)
    #print()
    gpr = make_gps(x=x, y=y, gp_name="MA")
    return gpr


def build_perf_model_from_spec_ssa(
    train_inputs_dict: dict[str, np.ndarray],
    train_targets_dict: dict[str, np.ndarray],
    num_samples: int,
    param_space: dict[str, str],
    metric_space: dict[str, str],
    obj_space: dict[str, str],
    edges: dict[str, list[str]],
    device,
) -> Dag:
    """
    build perf_dag from given spec (use sample average posterior)

    Core Args:
        param_space: key: param name - val: `categorical` or `continuous`
        metric_space: key: metric name - val: no meaning
        obj_space: key: obj name - val: no meaning
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
        obj_space, node_order)
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
            dag.register_metric(node, [i for i in reversed_edge[node]])
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
                                      edges: dict[str,
                                                  list[str]], device) -> Dag:
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
        obj_space, node_order)
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
            dag.register_metric(node, [i for i in reversed_edge[node]])
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
        train_inputs.append(torch.tensor(train_inputs_dict[node], dtype=dtype))

    # NOTE: if train_targets_dict has extra metrics that doesn't exist in performance model, they are not considerd
    for node in node_order:
        if node in param_space:
            continue
        elif node in metric_space or node in obj_space:
            train_target_names.append(node)
            train_targets.append(
                torch.tensor(train_targets_dict[node], dtype=dtype))
        else:
            raise RuntimeError(
                "node not in param_space or metric_space or obj_space")

    # format tensor, must be consistent with Dag's signature shape
    # NOTE: after the transpose, [q, dim]
    train_inputs = torch.stack(train_inputs).T
    train_targets = torch.stack(train_targets).T
    in_dim = len(train_input_names)
    target_dim = len(train_target_names)
    train_inputs = train_inputs.reshape(1, -1, in_dim)
    train_targets = train_targets.reshape(1, -1, target_dim)

    return train_input_names, train_target_names, train_inputs, train_targets


def standard_dict(input_dict, standardisation):
    dict_ = deepcopy(input_dict)
    if standardisation:
        for k, v in dict_.items():
            # StandardScaler, MinMaxScaler
            tmp = MinMaxScaler().fit_transform(v.reshape(-1, 1))
            dict_[k] = tmp.reshape(-1)
    return dict_


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
