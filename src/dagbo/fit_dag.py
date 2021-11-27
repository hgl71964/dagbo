import torch
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from pyro.infer import NUTS, MCMC
from botorch.fit import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch
from torch.optim import Adam
from .node import Node
from .dag import Dag
from typing import Any, Callable


def get_pyro_model(model: Node, mll: ExactMarginalLogLikelihood):
    def pyro_model(X, y):
        model.pyro_sample_from_prior()
        output = model(X)
        mll.pyro_factor(output, y)  # loss
        return y

    return pyro_model


def fit_node_with_mcmc(model: Node, num_samples: int, warmup_steps: int,
                       **kwargs: Any) -> None:
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()
    nuts = NUTS(get_pyro_model(model, mll))
    mcmc = MCMC(nuts, num_samples, warmup_steps)
    mcmc.run(*model.train_inputs, model.train_targets)
    # store the samples in the node by calling "pyro_load_from_samples"
    # this follows the same pattern as the torch / scipy optimisers which set the parameter values equal to their optimised value
    # the only difference is that they are set to a batch of values, one for each MCMC value
    model.pyro_load_from_samples(mcmc.get_samples())


def fit_node_with_torch(model: Node, **kwargs: Any) -> None:
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()
    fit_gpytorch_model(mll, fit_gpytorch_torch)


def fit_node_with_scipy(model: Node, **kwargs: Any) -> None:

    before = 0
    after = 0
    train_x = kwargs.get("train_x", None)
    train_y = kwargs.get("train_y", None)
    verbose = kwargs.get("verbose", False)
    if train_x is None or train_y is None:
        raise RuntimeError("no data to show fit")

    # before fit
    # nn.Module will set children (model and likelihood) to train too
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()
    with torch.no_grad():
        output = model(train_x)
        loss = -mll(output, train_y)
        before = loss.item()

    # fit
    fit_gpytorch_model(mll)  # by default it fits with scipy, so L-BFGS-B

    # after fit
    mll.train()  # fit_gpytorch_model will turn into eval()
    with torch.no_grad():
        output = model(train_x)
        loss = -mll(output, train_y)
        after = loss.item()
    if verbose:
        print(
            f"Neg log-likelihood before fit: {before:.2f} - after fit: {after:.2f}"
        )
    mll.eval()


def fit_node_with_adam(model: Node, **kwargs: Any) -> None:

    model.train()
    model.likelihood.train()
    iteration = kwargs.get("iteration", 64)
    lr = kwargs.get("lr", 0.1)
    verbose = kwargs.get("verbose", False)
    optimizer = Adam(model.parameters(), lr=lr)

    # get train data
    train_x = kwargs.get("train_x", None)
    train_y = kwargs.get("train_y", None)
    if train_x is None or train_y is None:
        raise RuntimeError("no data to train with ADAM")

    if verbose:
        print("data:")
        print(train_x)
        print(train_x.shape)
        print(train_y)
        print(train_y.shape)

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()

    for i in range(iteration):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        if verbose:
            print('Iter %d/%d - Loss: %.3f' % (
                i + 1,
                iteration,
                loss.item(),
            ))
        optimizer.step()
    mll.eval()


def fit_dag(dag_model: Dag,
            node_optimizer: Callable[[Node, Any], None] = fit_node_with_scipy,
            verbose: bool = False,
            **kwargs: Any) -> None:
    """"
    The training data of each node must have been saved to dag_model when instantiation

    Args:
        dag_model (Dag): the Dag model
        node_optimizer (Callable[[Node, Any], None], optional): Defaults to fit_node_with_scipy.
        verbose (bool, optional): print much more info during fitting. Defaults to False.
    """
    #for node in dag_model.nodes_output_order(): deprecated this order
    for node in dag_model.nodes_dag_order():
        if verbose:
            print("fitting node: ", node.output_name)

        # find saved tensor
        input_names = node.input_names
        node_name = node.output_name
        kwargs["train_x"], kwargs["train_y"] = dag_model.prepare_node_data(
            node_name, input_names)

        # fit
        kwargs["verbose"] = verbose
        node_optimizer(node, **kwargs)
