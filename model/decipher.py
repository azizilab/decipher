from dataclasses import dataclass
from typing import Union

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.nn.functional import softmax, softplus
import torch.utils.data
from model.module import ConditionalDenseNN


@dataclass(unsafe_hash=True)
class DecipherConfig:
    z_dim: int = 10
    v_dim: int = 2
    v_to_z_layers: tuple = (64,)
    z_to_x_layers: tuple = tuple()

    beta: float = 1.0
    seed: int = 0

    minibatch_rescaling: float = 1.0
    learning_rate: float = 5e-3

    prior: str = "normal"


class Decipher(nn.Module):
    """Decipher model for single-cell data.

    Parameters
    ----------
    genes_dim : int
        Number of genes in the dataset.
    decipher_config : DecipherConfig or dict
        Configuration for the decipher model.
    """

    def __init__(
        self,
        genes_dim,
        decipher_config: Union[DecipherConfig, dict] = DecipherConfig(),
    ):
        super().__init__()
        self.genes_dim = genes_dim
        if type(decipher_config) == dict:
            decipher_config = DecipherConfig(**decipher_config)

        self.prior = decipher_config.prior
        self.z_dim = decipher_config.z_dim
        self.v_sim = decipher_config.v_dim

        self.minibatch_rescaling = decipher_config.minibatch_rescaling
        self.decoder_v_to_z = ConditionalDenseNN(
            self.v_sim,
            decipher_config.v_to_z_layers,
            [self.z_dim, self.z_dim],
        )
        self.decoder_z_to_x = ConditionalDenseNN(
            self.z_dim, decipher_config.z_to_x_layers, [self.genes_dim]
        )
        self.encoder_x_to_z = ConditionalDenseNN(self.genes_dim, [128], [self.z_dim, self.z_dim])
        self.encoder_zx_to_v = ConditionalDenseNN(
            self.genes_dim + self.z_dim,
            [128],
            [self.v_sim, self.v_sim],
        )

        self._epsilon = 5e-3
        self.beta = decipher_config.beta
        self.decipher_config = decipher_config

        self.theta = None
        print("V5")

    def model(self, x, context=None):
        pyro.module("decipher", self)

        self.theta = pyro.param(
            "inverse_dispersion",
            1.0 * x.new_ones(self.genes_dim),
            constraint=constraints.positive,
        )

        with pyro.plate("batch", len(x)), poutine.scale(scale=self.minibatch_rescaling):
            with poutine.scale(scale=self.beta):
                if self.prior == "normal":
                    v = pyro.sample("v", dist.Normal(0, x.new_ones(self.v_sim)).to_event(1))
                elif self.prior == "gamma":
                    v = pyro.sample("v", dist.Gamma(0.3, x.new_ones(self.v_sim) * 0.8).to_event(1))
                else:
                    raise ValueError("Invalid prior, must be normal or gamma")

            z_loc, z_scale = self.decoder_v_to_z(v, context=context)
            z_scale = softplus(z_scale)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            mu = self.decoder_z_to_x(z, context=context)
            mu = softmax(mu, dim=-1)
            library_size = x.sum(axis=-1, keepdim=True)
            # Parametrization of Negative Binomial by the mean and inverse dispersion
            # See https://github.com/pytorch/pytorch/issues/42449
            # noinspection PyTypeChecker
            logit = torch.log(library_size * mu + self._epsilon) - torch.log(
                self.theta + self._epsilon
            )
            # noinspection PyUnresolvedReferences
            x_dist = dist.NegativeBinomial(total_count=self.theta, logits=logit)
            pyro.sample("x", x_dist.to_event(1), obs=x)

    def guide(self, x, context=None):
        pyro.module("decipher", self)
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.minibatch_rescaling):
            x = torch.log1p(x)

            z_loc, z_scale = self.encoder_x_to_z(x, context=context)
            z_scale = softplus(z_scale)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            zx = torch.cat([z, x], dim=-1)
            p_loc, p_scale = self.encoder_zx_to_v(zx, context=context)
            p_scale = softplus(p_scale)
            with poutine.scale(scale=self.beta):
                if self.prior == "gamma":
                    pyro.sample("v", dist.Gamma(softplus(p_loc), p_scale).to_event(1))
                elif self.prior == "normal" or self.prior == "student-normal":
                    pyro.sample("v", dist.Normal(p_loc, p_scale).to_event(1))
                else:
                    raise ValueError("Invalid prior, must be normal or gamma")
        return z_loc, p_loc

    def impute_data(self, x):
        z_loc, _ = self.guide(x)
        mu = self.decoder_z_to_x(z_loc)
        mu = softmax(mu, dim=-1)
        library_size = x.sum(axis=-1, keepdim=True)
        return library_size * mu


def make_data_loader_from_adata(adata, batch_size=64, context_discrete_keys=None):
    genes = torch.FloatTensor(adata.X.todense())
    params = [genes]
    context_tensors = []
    if context_discrete_keys is None:
        context_discrete_keys = []

    for key in context_discrete_keys:
        t = torch.IntTensor(adata.obs[key].astype("category").cat.codes.values).long()
        encoded = torch.nn.functional.one_hot(t).float()
        context_tensors.append(encoded)

    if context_tensors:
        context = torch.cat(context_tensors, dim=-1)
        params.append(context)

    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*params),
        batch_size=batch_size,
        shuffle=True,
    )
