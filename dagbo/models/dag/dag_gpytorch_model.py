import logging
import torch
from torch import Tensor
from typing import Any, Union
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

        # create multiple posteriors at identical input points
        # use multiple identical batch to represent i.i.d sample from posterior
        # expanded_X [num_sample, batch_size, q, d]
        original_shape = X.shape
        expanded_X = X.unsqueeze(dim=0).expand(self.num_samples,
                                               *original_shape).to(self.device)
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

        #print()
        #print("X::: ", X.shape)
        #print(X)
        #print("mvn:::")
        #print(mvn)
        #print(mvn.loc)
        #print()

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


class direct_DagGPyTorchModel(GPyTorchModel):
    """this impl direct average over sample dimenisons
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

        # create multiple posteriors at identical input points
        # use multiple identical batch to represent i.i.d sample from posterior
        # expanded_X [num_sample, batch_size, q, d]
        original_shape = X.shape
        expanded_X = X.unsqueeze(dim=0).expand(self.num_samples,
                                               *original_shape).to(self.device)
        #expanded_X = X

        # DAG's forward
        with gpt_posterior_settings():
            mvn = self(expanded_X)
            if observation_noise is not False:
                # TODO this add likelihood, which is a mapping from f to y
                raise NotImplementedError(
                    "Observation noise is not yet supported for DagGPyTorch models."
                )
        # GPyTorchPosterior support both MultitaskMultivariateNormal and MultivariateNormal
        if isinstance(mvn, MultivariateNormal):
            # mvn loc: [num_samples, batch_shape, q, num_nodes]
            gpytorch_mvn = MultivariateNormal(
                mvn.loc.mean(0),
                mvn.covariance_matrix.mean(0))  # take ave. along samples dim
            posterior = GPyTorchPosterior(mvn=gpytorch_mvn)
        else:
            # build fake gp output in the case of the obj node being a sum node
            cov_matrix = torch.eye(1) * 1e-6  # it is single obj so 1X1
            cov_matrix = cov_matrix.to(mvn)  # put to device & dtype
            gpytorch_mvn = MultivariateNormal(
                mvn.mean(0),
                cov_matrix)
            posterior = GPyTorchPosterior(mvn=gpytorch_mvn)

        #print()
        #print("X::: ", X.shape)
        #print(X)
        #print("mvn:::")
        #print(mvn)
        #print(mvn.loc)
        #print()
        if hasattr(self, "outcome_transform"):
            # posterior = self.outcome_transform.untransform_posterior(posterior)
            raise RuntimeError("does not support outcome_transform atm")

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
