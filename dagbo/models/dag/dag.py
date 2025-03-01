import logging
from copy import deepcopy
from torch import Size
from torch import Tensor
from typing import Iterator, Optional, Union
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from gpytorch.means.mean import Mean
from gpytorch.module import Module
from dagbo.utils.tensor_dict_conversions import pack_to_tensor, unpack_to_dict
from .node import Node, SingleTaskGP_Node, Sum_Node


class Dag(Module):
    """
    The DAG is a GPyTorch model with a directed, acyclic graph of sub-models. To
    create a DAG: subclass this class and implement define_dag using
    register_input and register_metric to "wire up" the DAG.

    DAG "inputs" are configurable parameters of the experiment. For example, when
    benchmarking a Java program we might configure the heap size and the number
    of CPUs.

    DAG "metrics" are the measurable outcomes of the experiment. For example,
    when benchmarking a Java program we might measure the total program time
    and also garbage-collection time.

    In the DAG model, all metrics are given their own sub-model which learns to
    predict the value of the metric. We can choose the inputs to each sub-model
    to be real inputs or to be other metrics.

    The sub-models are trained independently. Using the running example, the GC
    time of the Java program will be trained separately to the total time of the
    program. I.e.:
    the GC time will be trained on data {x: (heap_size, num_cpus), y: gc_time}
    and total time will be trained on data {x: (num_cpus, gc_time), y: total_time}

    During prediction, all sub-models are dependent on their inputs. I.e.:
    GC time uses x=(heap_size, num_cpus) to predict gc_time
    then total time uses y=(num_cpus, gc_time) to predict total_time
    """
    def __init__(self, train_input_names: list[str],
                 train_target_names: list[str], train_inputs: Tensor,
                 train_targets: Tensor, device):
        """
        Args:
            train_input_names: a d-length list of the names of each input
                defining the order of the inputs in the innermost
                dimension of train_inputs.
            train_target_names: a m-length list of the names of each target
                defining the order of the inputs in the innermost
                dimension of train_targets.
            train_inputs: A batch_shape*q*d-dim Tensor of the training inputs
                The innermost dim, dim=-1, must follow the same order as
                train_input_names as this is how the data is split up around
                the DAG. q is the number of available data point, and d is the
                dimension of BO's input space
            train_targets: A batch_shape*q*m-dim Tensor of the training targets
                The innermost dim, dim=-1, must follow the same order as
                train_targets_names as this is how the data is split up around
                the DAG. q is the number of available data point, and m is the
                dimension of BO's output space
        """
        super().__init__()
        self._check_valid_input(train_input_names, train_target_names,
                                train_inputs, train_targets)

        batch_shape = train_inputs.shape[:-2]
        assert batch_shape == Size([1]), f"get batch shape {batch_shape}"
        self.input_names = train_input_names
        self.target_names = train_target_names

        # data is explicitly un-batched in DAG since it will be split up for each sub-model
        # it will then be re-batched before adding it to the submodel
        self.train_inputs_name2tensor_mapping = unpack_to_dict(
            self.input_names, train_inputs)
        self.train_targets_name2tensor_mapping = unpack_to_dict(
            self.target_names, train_targets)

        # nodes themselves will be accessed using self.named_children
        self.registered_input_names = []
        self.registered_target_names = []
        self.define_dag(batch_shape)

        # tensor dtype and device conversion
        self.to(train_inputs)
        self.device = device

    """
    -------------- materialise DAG --------------
    """

    def define_dag(self, batch_shape: Size) -> None:
        """
        Must be implemented in subclass
        Creates the nodes and edges of the DAG
        """
        raise NotImplementedError

    def register_input(self, name: str) -> str:
        self._error_missing_names([name])
        self.registered_input_names.append(name)
        return name

    def register_metric(
            self,
            name: str,
            children: list[str],
            mean: Optional[Mean] = None,
            covar: Optional[Kernel] = None,
            likelihood: Optional[_GaussianLikelihoodBase] = None) -> str:
        """
        children is un-ordered
        """
        self._error_missing_names([name] + children)
        self._error_unregistered_inputs(children, name)

        # find saved tensor
        # batch gp inference : see https://docs.gpytorch.ai/en/stable/examples/08_Advanced_Usage/Simple_Batch_Mode_GP_Regression.html
        # X.shape = [batch-size, q, dim]
        # y.shape = [batch-size, q]  y's output dim is one
        X, y = self.prepare_node_data(name, children)
        self._check_init_metric_data(name, X, y)
        #print("dim: ", X.shape, y.shape)

        # instantial node
        X, y = deepcopy(X), deepcopy(y)
        #node = Node(children, name, X, y, mean, covar, likelihood)
        node = SingleTaskGP_Node(children, name, X, y, mean, covar, likelihood)
        self.add_module(
            name,
            node)  # nn.Module's method, keep a mapping dict[name, Module]
        self.registered_target_names.append(node.output_name)
        return name

    def register_normal_metric(
            self,
            name: str,
            children: list[str],
            mean: Optional[Mean] = None,
            covar: Optional[Kernel] = None,
            likelihood: Optional[_GaussianLikelihoodBase] = None) -> str:
        """
        children is un-ordered
        """
        self._error_missing_names([name] + children)
        self._error_unregistered_inputs(children, name)

        X, y = self.prepare_node_data(name, children)
        self._check_init_metric_data(name, X, y)

        # instantial node
        X, y = deepcopy(X), deepcopy(y)
        #node = Node(children, name, X, y, mean, covar, likelihood)
        node = Sum_Node(children, name, X, y, mean, covar, likelihood)
        self.add_module(
            name,
            node)  # nn.Module's method, keep a mapping dict[name, Module]
        self.registered_target_names.append(node.output_name)
        return name

    def prepare_node_data(self, name: str,
                          children: list[str]) -> tuple[Tensor, Tensor]:

        # find saved tensor
        X_from_inputs = {
            k: v
            for k, v in self.train_inputs_name2tensor_mapping.items()
            if k in children
        }
        X_from_outputs = {  # a metric could be a child of this node
            k: v
            for k, v in self.train_targets_name2tensor_mapping.items()
            if k in children
        }
        X = pack_to_tensor(children, {**X_from_inputs, **X_from_outputs})
        y = self.train_targets_name2tensor_mapping[name]
        self._check_init_metric_data(name, X, y)
        return X, y

    """
    -------------- forward and backward on DAG --------------
    """

    # the original forward implementation
    def forward(
        self, tensor_inputs: Tensor
    ) -> Union[MultivariateNormal, MultitaskMultivariateNormal]:
        """
        This is only used for prediction, since the individual nodes
        are trained independently
        Args:
            tensor_inputs: batch_shape*q*d-dim tensor
        """
        # since the nodes must be registered in topological order FIXME: user may not know
        #   then we can do the predictions in the same order and use
        #   tensor_inputs_dict to store them
        # also need to pack into tensors before passing to sub-models
        # FIXME, add support for MOBO? so multiple sink nodes?

        #print("DAG forwarding is called")
        #print(tensor_inputs.shape)

        #tensor_inputs_dict = unpack_to_dict(self.registered_input_names,
        #                                    tensor_inputs)
        #node_dict = {}

        ## MUST traverse in topological order
        #for node in self.nodes_dag_order():

        #    # prepare input to each node
        #    node_inputs_dict = {
        #        k: v
        #        for k, v in tensor_inputs_dict.items() if k in node.input_names
        #    }
        #    node_inputs = pack_to_tensor(node.input_names, node_inputs_dict)

        #    # make prediction via GP
        #    mvn = node(node_inputs)

        #    #print("node: ", node.output_name)
        #    #print(node_inputs.shape)
        #    #print("mvn:")
        #    #print(mvn)
        #    #print(mvn.event_shape, mvn.batch_shape)
        #    #if node.output_name == "z2":
        #    #    print(mvn.loc)  # can verify identical mvn

        #    node_dict[node.output_name] = mvn
        #    prediction = mvn.rsample()
        #    tensor_inputs_dict[node.output_name] = prediction

        ## aggregate posterior from all nodes/metrics
        #if len(self.registered_target_names) > 1:
        #    # mvns must be in the expected output order
        #    mvns = [
        #        node_dict[metric] for metric in self.registered_target_names
        #    ]
        #    return MultitaskMultivariateNormal.from_independent_mvns(mvns)
        ## cannot have 1 task in MultitaskMultivariateNormal
        #else:
        #    return node_dict[self.registered_target_names[0]]
        raise NotImplementedError

    """
    -------------- ordering --------------
    """

    def _nodes_order(self, order) -> Iterator[Node]:
        """Returns: iterator over DAG's nodes in `order` order"""
        for name in order:
            yield getattr(self, name)

    def nodes_dag_order(self) -> Iterator[Node]:
        """Returns: iterator over DAG's nodes in the order specified in define_dag"""
        return self._nodes_order(self.registered_target_names)

    """
    -------------- checking --------------
    """

    def _error_missing_names(self, names):
        missing_names = set(names).difference(self.input_names).difference(
            self.target_names)
        if missing_names:
            raise NameError(
                str(missing_names) +
                " defined in DAG but not declared in train_input_names or train_target_names."
            )

    def _error_unregistered_inputs(self, input_names, output_name):
        unregisted_inputs = set(input_names).difference(
            self.registered_input_names).difference(
                self.registered_target_names)
        if unregisted_inputs:
            raise NameError(
                str(unregisted_inputs) + " defined as input to " +
                output_name + " before being registered.")

    def _check_valid_input(self, train_input_names: list[str],
                           train_target_names: list[str], train_inputs: Tensor,
                           train_targets: Tensor):
        if len(set(train_input_names)) != len(train_input_names):
            raise RuntimeError("train_input_names has duplicated name")

        if len(set(train_target_names)) != len(train_target_names):
            raise RuntimeError("train_target_names has duplicated name")

        if len(train_inputs.shape) != 3 or len(train_targets.shape) != 3:
            raise RuntimeError(
                "train_inputs and train_targets must be 3 dimensional tensor")

        if train_inputs.shape[-1] != len(train_input_names):
            raise RuntimeError("input dim != input name len")

        if train_targets.shape[-1] != len(train_target_names):
            raise RuntimeError("target dim != target name len")

        if train_inputs.shape[0] != 1 or train_targets.shape[0] != 1:
            raise RuntimeError(
                f"""instantiate DAG does not allow batch shape > 1,
                however the batch dimension must be kept
                for later acquisition function optimisation""")

        if train_inputs.shape[1] != train_targets.shape[1]:
            q1, q2 = train_inputs.shape[1], train_targets.shape[1]
            raise RuntimeError(
                f"q in train_input is {q1} but in train_targets is {q2}, and they should be equal"
            )

    def _check_init_metric_data(self, name, X, y):
        batch_size, q, _ = X.shape
        batch_size2, q2 = y.shape
        if len(y.shape) != 2:
            raise RuntimeError(
                f"node {name} only support 1 output for now, but output data shape {y.shape}"
            )
        if batch_size != batch_size2:
            raise RuntimeError(
                f"node {name} has input batch size {batch_size} but output batch_size {batch_size2}"
            )
        if q != q2:
            raise RuntimeError(
                f"node {name} has input data points {q} but output data point {q2}"
            )


class SO_Dag(Dag):
    """
    Single objective Dag:
        a Dag only return sink node's multi-variate normal distribution

    Args:
        Dag ([type]): see above
    """
    def __init__(self, train_input_names: list[str],
                 train_target_names: list[str], train_inputs: Tensor,
                 train_targets: Tensor, device):
        super().__init__(train_input_names, train_target_names, train_inputs,
                         train_targets, device)

        # NOTE: this will be used by Botorch's API
        self._num_outputs = 1

    def define_dag(self, batch_shape: Size) -> None:
        raise NotImplementedError

    def forward(self, tensor_inputs: Tensor) -> MultivariateNormal:
        """
        Args:
            tensor_inputs: batch_shape*q*d-dim tensor
        """
        #tensor_inputs_dict = unpack_to_dict(self.registered_input_names,
        #                                    tensor_inputs)
        tensor_inputs_dict = unpack_to_dict(self.input_names, tensor_inputs)
        node_dict = {}
        sink_node_name = None

        # ensure traverse in topological order
        for node in self.nodes_dag_order():

            # prepare input to each node
            node_inputs_dict = {
                k: v
                for k, v in tensor_inputs_dict.items() if k in node.input_names
            }
            node_inputs = pack_to_tensor(node.input_names, node_inputs_dict)

            # make prediction via GP
            mvn = node(node_inputs)
            # XXX likelihood or not
            #like_mvn = node.likelihood(mvn, node_inputs)
            like_mvn = mvn

            node_dict[node.output_name] = like_mvn
            prediction = like_mvn.rsample()
            #print()
            #print("rsample size::")
            #print(node.output_name)
            #print(node_inputs.shape)
            #print(prediction.shape)
            #print(node_inputs)
            #print(like_mvn.loc)
            tensor_inputs_dict[node.output_name] = prediction
            sink_node_name = node.output_name

        return node_dict[sink_node_name]


class lazy_SO_Dag(Dag):
    """
    Lazily initialized single objective Dag:
        define_dag is postponed to execute

    Args:
        Dag ([type]): see above
    """
    def __init__(self, train_input_names: list[str],
                 train_target_names: list[str], train_inputs: Tensor,
                 train_targets: Tensor, device):
        super().__init__(train_input_names, train_target_names, train_inputs,
                         train_targets, device)

        # NOTE: this will be used by Botorch's API
        self._num_outputs = 1

    def define_dag(self, batch_shape: Size) -> None:
        pass

    def forward(self, tensor_inputs: Tensor) -> MultivariateNormal:
        """
        Args:
            tensor_inputs: batch_shape*q*d-dim tensor,
                    its d-dim is ordered
            this order should be maintained by bounds
        """
        # will be update as traversal on-the-fly
        #tensor_inputs_dict = unpack_to_dict(self.registered_input_names,
        #                                    tensor_inputs)
        tensor_inputs_dict = unpack_to_dict(self.input_names, tensor_inputs)
        node_dict = {}
        sink_node_name = None

        # ensure traverse in topological order
        for node in self.nodes_dag_order():

            # prepare input to each node
            node_inputs_dict = {
                k: v
                for k, v in tensor_inputs_dict.items() if k in node.input_names
            }
            # node.input_names = register_metric's children, so order is preserved
            node_inputs = pack_to_tensor(node.input_names, node_inputs_dict)

            # make prediction via GP, XXX likelihood or not
            mvn = node(node_inputs)
            #like_mvn = node.likelihood(mvn, node_inputs)
            like_mvn = mvn
            node_dict[node.output_name] = like_mvn

            # append prediction to tensor_inputs_dict
            if isinstance(mvn, MultivariateNormal):
                prediction = like_mvn.rsample()
            else:
                prediction = mvn
            tensor_inputs_dict[node.output_name] = prediction
            sink_node_name = node.output_name

        return node_dict[sink_node_name]
