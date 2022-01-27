from dagbo.dag import lazy_SO_Dag, Dag


def build_perf_model_from_text(train_input_names: list[str],
                               train_target_names: list[str],
                               train_inputs: Tensor, train_targets: Tensor,
                               num_samples: int, param_space: dict,
                               metric_space: dict, obj_space: dict,
                               edges: dict) -> Dag:

    dag = lazy_SO_Dag(train_input_names, train_target_names, train_inputs,
                      train_targets, num_samples)

    # input space, TODO address for categorical variable in the future
    for input_name in list(param_space.keys()):
        dag.register_input(input_name)

    # TODO
    z_1 = self.register_metric("z1", [x_1, x_2])
    z_2 = self.register_metric("z2", [x_3])

    y = self.register_metric("y", [z_1, z_2])
    return dag


def get_dag_topological_order(param_space: dict, metric_space: dict,
                              obj_space: dict, edges: dict) -> list[str]:
    """
    return a list of nodes following the topological order of the DAG
    """
    sink = list(obj_space.keys())
    assert (len(sink) == 1), "does not support multiple objective for now"

    reversed_edge = find_inverse_edges(edges)
    input_space = list(param_space.keys())
    intermediate_space = list(metric_space.keys())

    # TODO

    return


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
