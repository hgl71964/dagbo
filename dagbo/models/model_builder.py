import botorch
import gpytorch
from torch import Tensor
from gpytorch.kernels.kernel import Kernel
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from dagbo.models.gp_factory import make_gps
from dagbo.utils.perf_model_utils import get_dag_topological_order, find_inverse_edges


def build_gp_from_spec(train_inputs_dict: dict[str, np.ndarray],
                       train_targets_dict: dict[str, np.ndarray],
                       param_space: dict[str, str],
                       metric_space: dict[str, str], obj_space: dict[str, str],
                       edges: dict[str, list[str]], standardisation: bool):

    node_order = get_dag_topological_order(obj_space, edges)

    ## standardisation
    train_inputs_dict_, train_targets_dict_ = standard_dict(
        train_inputs_dict,
        standardisation), standard_dict(train_targets_dict, standardisation)

    ##
    train_input_names, train_target_names, train_inputs, train_targets = build_input_by_topological_order(
        train_inputs_dict_, train_targets_dict_, param_space, metric_space,
        obj_space, node_order)

    # TODO

    gpr = make_gps(x=x, y=y, gp_name="MA")
    return gpr


def build_perf_model_from_spec_ssa(train_inputs_dict: dict[str, np.ndarray],
                                   train_targets_dict: dict[str, np.ndarray],
                                   num_samples: int, param_space: dict[str,
                                                                       str],
                                   metric_space: dict[str, str],
                                   obj_space: dict[str, str],
                                   edges: dict[str, list[str]],
                                   standardisation: bool) -> Dag:
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
                     train_targets: Tensor, num_samples: int):
            super().__init__(train_input_names, train_target_names,
                             train_inputs, train_targets)
            self.num_samples = num_samples

    # build
    reversed_edge = find_inverse_edges(edges)
    node_order = get_dag_topological_order(obj_space, edges)

    ## standardisation
    train_inputs_dict_, train_targets_dict_ = standard_dict(
        train_inputs_dict,
        standardisation), standard_dict(train_targets_dict, standardisation)

    ##
    train_input_names, train_target_names, train_inputs, train_targets = build_input_by_topological_order(
        train_inputs_dict_, train_targets_dict_, param_space, metric_space,
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


def build_perf_model_from_spec_direct(train_inputs_dict: dict[str, np.ndarray],
                                      train_targets_dict: dict[str,
                                                               np.ndarray],
                                      num_samples: int, param_space: dict[str,
                                                                          str],
                                      metric_space: dict[str, str],
                                      obj_space: dict[str, str],
                                      edges: dict[str, list[str]],
                                      standardisation: bool) -> Dag:
    """
    use approx. posterior
    """
    class perf_DAG(lazy_SO_Dag, direct_DagGPyTorchModel):
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
    ## standardisation
    train_inputs_dict_, train_targets_dict_ = standard_dict(
        train_inputs_dict,
        standardisation), standard_dict(train_targets_dict, standardisation)

    ##
    train_input_names, train_target_names, train_inputs, train_targets = build_input_by_topological_order(
        train_inputs_dict_, train_targets_dict_, param_space, metric_space,
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
    for node in node_order:
        if node in param_space:
            train_input_names.append(node)
            train_inputs.append(
                torch.tensor(train_inputs_dict[node], dtype=dtype))
        elif node in metric_space or node in obj_space:
            train_target_names.append(node)
            train_targets.append(
                torch.tensor(train_targets_dict[node], dtype=dtype))
        else:
            raise RuntimeError(
                "node not in param_space or metric_space or obj_space")

    # format tensor, must be consistent with Dag's signature shape
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
            tmp = StandardScaler().fit_transform(v.reshape(-1, 1))
            dict_[k] = tmp.reshape(-1)
    return dict_
