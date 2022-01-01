import logging
from torch import Tensor
from typing import Any, list, Union
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils import gpt_posterior_settings
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal

from .sample_average_posterior import SampleAveragePosterior, SampleAveragePosterior_v2


class DagGPyTorchModel(GPyTorchModel):
    """this impl the posterior methods to generate samples
    Samples are generated via a `SampleAveragePosterior` fashion

    Args:
        GPyTorchModel ([type]): BoTorch's model with a posterior methods

    Raises:
        NotImplementedError: [description]
        RuntimeError: [description]
        NotImplementedError: [description]
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """
    num_samples: int

    def posterior(self,
                  X: Tensor,
                  observation_noise: Union[bool, Tensor] = False,
                  **kwargs: Any) -> GPyTorchPosterior:
        """Computes the posterior over model outputs at the provided points.
        acquisition function will call this to generate samples

        When calling botorch's optimize_acqf, the batch dimension of X
                                                    is the number of restarts          

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`).

        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """
        self.eval()  # make sure model is in eval mode

        verbose = kwargs.get("verbose", False)

        # create multiple posteriors at identical input points
        # use multiple identical batch to represent i.i.d sample from posterior
        # expanded_X [num_sample, batch_size, q, d]
        original_shape = X.shape
        expanded_X = X.unsqueeze(dim=0).expand(self.num_samples,
                                               *original_shape)
        # DAG's forward
        with gpt_posterior_settings():
            mvn = self(expanded_X)
            if observation_noise is not False:
                # TODO this add likelihood, which is a mapping from f to y
                raise NotImplementedError(
                    "Observation noise is not yet supported for DagGPyTorch models."
                )
        # GPyTorchPosterior support both MultitaskMultivariateNormal and MultivariateNormal
        # mvn: [num_samples, batch_shape, q, num_nodes]
        posterior = GPyTorchPosterior(mvn=mvn)
        if verbose:
            logging.info("DAG's posterior: ")
            print("expanded_X: ", expanded_X.shape)
            print("mvn: ", mvn, mvn.loc.shape)
            print("posterior: ", posterior.event_shape, posterior.mean.shape)
        if hasattr(self, "outcome_transform"):
            # posterior = self.outcome_transform.untransform_posterior(posterior)
            raise RuntimeError("does not support outcome_transform atm")

        # SampleAverage uses a multi-variate normal distribution to approximate complex posterior
        #posterior = SampleAveragePosterior.from_gpytorch_posterior(posterior)
        posterior = SampleAveragePosterior_v2.from_gpytorch_posterior(
            posterior)
        return posterior

    def condition_on_observations(self, X: Tensor, Y: Tensor,
                                  **kwargs: Any) -> Model:
        raise NotImplementedError(
            "Condition on observations is not yet supported for DagGPyTorch models"
        )

    def subset_output(self, idcs: list[int]) -> Model:
        raise NotImplementedError(
            "Condition on observations is not yet supported for DagGPyTorch models"
        )
