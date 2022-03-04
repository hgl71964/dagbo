import botorch
import gpytorch
from torch import Tensor
from gpytorch.kernels.kernel import Kernel
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from dagbo.utils.perf_model_utils import get_dag_topological_order, build_input_by_topological_order


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
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5))
    return kernel


def fit_gpr(model: SingleTaskGP) -> None:
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()
    fit_gpytorch_model(mll)  # by default it fits with scipy, so L-BFGS-B
    return None

def build_gp_from_spec(train_inputs_dict: dict[str, np.ndarray]],
                       train_targets_dict: dict[str, np.ndarray]],
                       param_space: dict[str, str], metric_space: dict[str,
                                                                       str],
                       obj_space: dict[str, str], edges: dict[str, list[str]],
                       standardisation: bool):

    node_order = get_dag_topological_order(obj_space, edges)

    # TODO standardisation
    if standardisation:
        pass
    else:
        pass

    train_input_names, train_target_names, train_inputs, train_targets = build_input_by_topological_order(
        train_inputs_dict, train_targets_dict, param_space, metric_space,
        obj_space, node_order)


    gpr = make_gps(x=x, y=y, gp_name="MA")
    return gpr
