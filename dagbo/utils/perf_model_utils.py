import numpy as np
import torch
from torch import Tensor
from ax import Experiment
from copy import deepcopy
from typing import Union
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dagbo.models.dag.dag import lazy_SO_Dag, Dag
from dagbo.models.dag.dag_gpytorch_model import DagGPyTorchModel, direct_DagGPyTorchModel


def get_dag_topological_order(obj_space: dict[str, str],
                              edges: dict[str, list[str]]) -> list[str]:
    """
    return a list of nodes following the topological order of the DAG

    Args:
        edges: DAG forward direction
    """
    sink = list(obj_space.keys())

    # for multiple sink, topological sort see leetcode: Course Schedule II
    assert (len(sink) == 1), "does not support multiple objective for now"
    reversed_edge = find_inverse_edges(edges)

    # init dag node
    sink_node = sink[0]
    order = []
    visited = set()

    def dfs(node, edges, visited, order):
        visited.add(node)
        no_circle = True
        if node in edges:
            for child in edges[node]:
                if child in visited:
                    return False
                elif child in order:
                    pass
                else:
                    no_circle = no_circle and dfs(child, edges, visited, order)
        visited.remove(node)
        order.append(node)
        return no_circle

    no_circle = dfs(sink_node, reversed_edge, visited, order)
    if not no_circle:
        raise RuntimeError("this dag has circle")

    return order


def find_inverse_edges(edges: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    given graphviz source,
        edges directions are defined in the forward order, i.e. from parameter -> metric -> objective

    To find the topological order, it is easier to get edges in the reversed direction
    """
    reversed_edge = {}
    for node in list(edges.keys()):
        for tgt in edges[node]:
            if tgt in reversed_edge:
                reversed_edge[tgt].append(node)
            else:
                reversed_edge[tgt] = [node]
    return reversed_edge
