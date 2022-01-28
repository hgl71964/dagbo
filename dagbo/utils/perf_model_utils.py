import torch
from torch import Tensor
from dagbo.dag import lazy_SO_Dag, Dag
from dagbo.dag_gpytorch_model import DagGPyTorchModel


def build_perf_model_from_spec(train_inputs_dict: dict[str, Tensor],
                               train_targets_dict: dict[str, Tensor],
                               num_samples: int, param_space: dict,
                               metric_space: dict, obj_space: dict,
                               edges: dict[str, list[str]]) -> Dag:
    """
    Core utils func to build perf_dag from given spec
    """
    class perf_DAG(lazy_SO_Dag, DagGPyTorchModel):
        """dynamically define dag
        """
        def __init__(self, train_input_names: list[str],
                     train_target_names: list[str], train_inputs: Tensor,
                     train_targets: Tensor, num_samples: int):
            super().__init__(train_input_names, train_target_names,
                             train_inputs, train_targets)
            self.num_samples = num_samples

    # build
    reversed_edge = find_inverse_edges(edges)
    node_order = get_dag_topological_order(obj_space, edges)
    train_input_names, train_target_names, train_inputs, train_targets = build_input_by_topological_order(
        train_inputs_dict, train_targets_dict, param_space, metric_space,
        obj_space, node_order)
    dag = perf_DAG(train_input_names, train_target_names, train_inputs,
                   train_targets, num_samples)

    # register nodes, MUST register in topological order
    # input space, TODO address for categorical variable in the future
    for node in node_order:
        if node in param_space:
            dag.register_input(node)
        elif node in metric_space or node in obj_space:
            dag.register_metric(node, [i for i in reversed_edge[node]])
        else:
            raise RuntimeError("unknown node")

    return dag


def build_input_by_topological_order(
        train_inputs_dict: dict[str, Tensor], train_targets_dict: dict[str,
                                                                       Tensor],
        param_space: dict, metric_space: dict, obj_space: dict,
        node_order: list[str]) -> tuple[list[str], list[str], Tensor, Tensor]:
    """
    node names and their corresponding tensor must match

    Assume:
        Tensor from dict is 1-dim, each element represents a sample
            and the len of Tensor must be equal
    """
    train_input_names, train_target_names = [], []
    train_inputs, train_targets = [], []
    for node in node_order:
        if node in param_space:
            train_input_names.append(node)
            train_inputs.append(train_inputs_dict[node])
        elif node in metric_space or node in obj_space:
            train_target_names.append(node)
            train_targets.append(train_targets_dict[node])

    # format tensor, must be consistent with Dag's signature shape
    train_inputs = torch.stack(train_inputs).T
    train_targets = torch.stack(train_targets).T
    in_dim = len(train_input_names)
    target_dim = len(train_target_names)
    train_inputs = train_inputs.reshape(1, -1, in_dim)
    train_targets = train_targets.reshape(1, -1, target_dim)

    return train_input_names, train_target_names, train_inputs, train_targets


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
