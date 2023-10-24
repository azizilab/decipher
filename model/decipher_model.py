from dataclasses import dataclass

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.nn.functional import softmax, softplus
from model.utils import ConditionalDenseNN


@dataclass(unsafe_hash=True)
class DecipherConfig:
    latent_dim: int = 10
    pre_latent_dim: int = 2
    scale_factor: float = 1.0
    prior: str = "normal"
    p_to_z_layers: tuple = (64,)
    z_to_x_layers: tuple = tuple()
    beta: float = 1.0

    learning_rate: float = 5e-3
    seed: int = 0
    context: tuple = tuple()


class Decipher(nn.Module):
    def __init__(
        self,
        genes_dim,
        context_dim=0,
        decipher_config=DecipherConfig(),
    ):
        self.genes_dim = genes_dim
        self.context_dim = context_dim

        self.prior = decipher_config.prior
        self.latent_dim = decipher_config.latent_dim
        self.pre_latent_dim = decipher_config.pre_latent_dim

        self.scale_factor = decipher_config.scale_factor

        super().__init__()

        self.decoder_p_to_z = ConditionalDenseNN(
            self.pre_latent_dim,
            decipher_config.p_to_z_layers,
            [self.latent_dim, self.latent_dim],
            self.context_dim,
        )
        self.decoder_z_to_x = ConditionalDenseNN(
            self.latent_dim, decipher_config.z_to_x_layers, [self.genes_dim], self.context_dim
        )
        self.encoder_x_to_z = ConditionalDenseNN(
            self.genes_dim, [128], [self.latent_dim, self.latent_dim], self.context_dim
        )
        self.encoder_zx_to_p = ConditionalDenseNN(
            self.genes_dim + self.latent_dim,
            [128],
            [self.pre_latent_dim, self.pre_latent_dim],
            self.context_dim,
        )

        self.epsilon = 5.0e-3
        self.beta = decipher_config.beta
        self.theta = None
        self.decipher_config = decipher_config
        print("V4")

    def model(self, x, context=None):
        pyro.module("cvi", self)

        # Inverse dispersion: see ScanVI pyro code
        theta = pyro.param(
            "inverse_dispersion",
            1.0 * x.new_ones(self.genes_dim),
            constraint=constraints.positive,
        )
        self.theta = theta

        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            # KL annealing might be useful
            with poutine.scale(scale=self.beta):
                if self.prior == "gamma":
                    p = pyro.sample(
                        "p", dist.Gamma(0.3, x.new_ones(self.pre_latent_dim) * 0.8).to_event(1)
                    )
                elif self.prior == "normal" or self.prior == "normal-student":
                    p = pyro.sample(
                        "p", dist.Normal(0, x.new_ones(self.pre_latent_dim)).to_event(1)
                    )
                elif self.prior == "student-normal":
                    p = pyro.sample(
                        "p", dist.StudentT(1, 0, x.new_ones(self.pre_latent_dim)).to_event(1)
                    )
                else:
                    raise ValueError(
                        "prior should be one of normal, gamma, student-normal, normal-student"
                    )

            z_loc, z_scale = self.decoder_p_to_z(p, context=context)
            z_scale = softplus(z_scale)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            mu = self.decoder_z_to_x(z, context=context)
            mu = softmax(mu, dim=-1)
            library_size = x.sum(axis=-1, keepdim=True)
            # See https://github.com/pytorch/pytorch/issues/42449 for Negative Binomial parametrization
            nb_logits = (library_size * mu + self.epsilon).log() - (theta + self.epsilon).log()
            x_dist = dist.NegativeBinomial(total_count=theta, logits=nb_logits)
            pyro.sample("x", x_dist.to_event(1), obs=x)

    def guide(self, x, context=None):
        pyro.module("cvi", self)
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            # x =  x / x.sum(axis=1, keepdim=True)*1000
            x = torch.log1p(x)

            z_loc, z_scale = self.encoder_x_to_z(x, context=context)
            z_scale = softplus(z_scale)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            zx = torch.cat([z, x], dim=-1)
            p_loc, p_scale = self.encoder_zx_to_p(zx, context=context)
            p_scale = softplus(p_scale)
            with poutine.scale(scale=self.beta):
                if self.prior == "gamma":
                    p = pyro.sample("p", dist.Gamma(softplus(p_loc), p_scale).to_event(1))
                elif self.prior == "normal" or self.prior == "student-normal":
                    p = pyro.sample("p", dist.Normal(p_loc, p_scale).to_event(1))
                elif self.prior == "normal-student":
                    p = pyro.sample("p", dist.StudentT(5, p_loc, p_scale).to_event(1))

        return z_loc, p_loc

    def impute_data(self, x):
        z_loc, _ = self.guide(x)
        mu = self.decoder_z_to_x(z_loc)
        mu = softmax(mu, dim=-1)
        library_size = x.sum(axis=-1, keepdim=True)
        return library_size * mu

    def reconstruct_from_prelatent_numpy(self, p, scale=1.0):
        z_loc, z_scale = self.decoder_p_to_z(p)
        mu = softmax(self.decoder_z_to_x(z_loc), dim=-1)
        return (mu * scale).detach().numpy(), z_loc.detach().numpy()


# def make_data_loader(observations, batch_size=64, context=None):
#     return torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(torch.FloatTensor(observations)),
#         batch_size=batch_size,
#         shuffle=True,
#     )


def make_data_loader_from_adata(adata, batch_size=64, context_discrete_keys=[]):
    genes = torch.FloatTensor(adata.X.todense())
    params = [genes]
    context_tensors = []

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
