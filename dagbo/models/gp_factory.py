import botorch
import gpytorch
from torch import Tensor

from gpytorch.kernels.kernel import Kernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils import gpt_posterior_settings

from dagbo.models.dag.node import Node, SingleTaskGP_Node


def make_gps(x: Tensor, y: Tensor, gp_name: str) -> SingleTaskGP:

    # noiseless modelling
    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #likelihood.noise = 1e-4
    #likelihood.noise_covar.raw_noise.requires_grad_(False)

    # get model
    #model = SingleTaskGP(x, y, likelihood)
    model = SingleTaskGP(x, y)

    # equip
    #model.likelihood = likelihood
    model.covar_module = make_kernels(gp_name)
    return model


def make_kernels(name: str) -> Kernel:
    if name == "SE":
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    elif name == "RQ":
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
    elif name == "MA":
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(
            nu=2.5, lengthscale_prior=GammaPrior(3.0, 6.0)),
                                              outputscale_prior=GammaPrior(
                                                  2.0, 0.15))
    return kernel


def fit_gpr(model: SingleTaskGP) -> None:
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()
    fit_gpytorch_model(mll)  # by default it fits with scipy, so L-BFGS-B
    return None


def make_node(x: Tensor, y: Tensor, gp_name: str):
    """
    for test purpose
        check if Node in Dag is a sound gp
    """
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
            posterior = GPyTorchPosterior(mvn=mvn)
            return posterior

    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    if len(y.shape) == 2:
        y = y.unsqueeze(0)
        y = y.squeeze(-1)

    #print("node input:")
    #print(x.shape)  # [batch_size, q, dim]
    #print(y.shape)  # [batch_size, q]
    #print()

    model = gp([f"x{i}" for i in range(20)], "final", x, y)
    model.covar = make_kernels(gp_name)
    return model


def make_SingleTaskGP_node(x: Tensor, y: Tensor, gp_name: str):
    """
    for test purpose
        if SingleTaskGP_Node in Dag is a sound gp
    """
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    if len(y.shape) == 2:
        y = y.unsqueeze(0)
        y = y.squeeze(-1)

    #print("node input:")
    #print(x.shape)  # [batch_size, q, dim]
    #print(y.shape)  # [batch_size, q]
    #print()
    model = SingleTaskGP_Node([f"x{i}" for i in range(20)], "final", x, y)
    model.covar_module = make_kernels(gp_name)
    return model
