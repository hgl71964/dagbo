import botorch
import gpytorch
from torch import Tensor
from gpytorch.kernels.kernel import Kernel
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


def make_gps(x: Tensor, y: Tensor, gp_name: str) -> SingleTaskGP:

    # noiseless modelling
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = 1e-4
    likelihood.noise_covar.raw_noise.requires_grad_(False)

    # get kernel
    model = SingleTaskGP(x, y, likelihood)

    # equip
    model.likelihood = likelihood
    model.covar_module = make_kernels(gp_name)
    return model


def make_kernels(name: str) -> Kernel:
    if name == "SE":
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    elif name == "RQ":
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
    elif name == "MA":
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5))
    return kernel


def fit_gpr(model: SingleTaskGP) -> None:
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()
    fit_gpytorch_model(mll)  # by default it fits with scipy, so L-BFGS-B
    return None
