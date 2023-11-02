import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule


class RealFunction(pyro.nn.PyroModule):
    def __init__(self, *hidden_dims, sigmoid_output=False):
        super(RealFunction, self).__init__()
        dimensions = [1, *hidden_dims, 1]
        self.sigmoid_output = sigmoid_output
        layers = []
        for in_features, out_features in zip(dimensions, dimensions[1:]):
            layer = pyro.nn.PyroModule[torch.nn.Linear](in_features, out_features)
            layer.weight = pyro.nn.PyroSample(
                dist.Normal(0.0, 1.0 / in_features ** 0.5)
                .expand(torch.Size([out_features, in_features]))
                .to_event(2)
            )
            layer.bias = pyro.nn.PyroSample(
                dist.Normal(0.0, 1.0 / in_features ** 0.5)
                .expand(torch.Size([out_features]))
                .to_event(1)
            )
            layers.append(layer)

        self.layers = layers
        for i in range(len(layers)):
            setattr(self, f"layer_{i}", layers[i])

        self.activation = torch.nn.Tanh()

    def forward(self, x):
        x = x.view((-1, 1))
        for i in range(len(self.layers) - 1):
            layer = getattr(self, f"layer_{i}")
            x = layer(x)
            x = self.activation(x)
        layer = getattr(self, f"layer_{len(self.layers) - 1}")
        x = layer(x)  # linear output
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class BasisDecomposition(torch.nn.Module):
    def __init__(self, K, n_genes, n_conditions=2, beta_prior=1.0, normalized_mode=True):
        super().__init__()
        self.K = K
        self.normalized_mode = normalized_mode

        self.trajectory_basis = pyro.nn.PyroModule[torch.nn.ModuleList](
            [RealFunction(32, 32, sigmoid_output=self.normalized_mode) for _ in range(self.K)]
        )
        self.n_conditions = n_conditions
        self.n_genes = n_genes
        self.K = K
        self.beta_prior = beta_prior

        self.return_basis = True
        self._last_basis = None
        self._last_patterns = None

    @property
    def gene_scales(self):
        return torch.exp(pyro.param("gene_scale", torch.zeros(self.n_conditions, self.n_genes)))

    @property
    def std(self):
        return torch.exp(pyro.param("std", torch.zeros(1)))

    def forward(self, times, data):
        basis = self.get_basis(times)

        if not self.normalized_mode:
            betas = pyro.sample(
                "beta",
                dist.Exponential(self.beta_prior)
                .expand([self.n_conditions, self.n_genes, self.K])
                .to_event(3),
            )
            std = 0.1
        else:
            betas = pyro.sample(
                "beta",
                dist.Dirichlet(torch.ones(self.K) * self.beta_prior)
                .expand(torch.Size([self.n_conditions, self.n_genes]))
                .to_event(2),
            )
            gene_scales = self.gene_scales.unsqueeze(-1)
            betas = betas * gene_scales
            std = self.std * gene_scales

        gene_patterns = torch.einsum("cgk, tk -> cgt", betas, basis)

        t_axis = pyro.plate("t", len(times), dim=-1)
        gene_axis = pyro.plate("g", self.n_genes, dim=-2)
        c_axis = pyro.plate("c", self.n_conditions, dim=-3)

        with c_axis, gene_axis, t_axis:
            pyro.sample("obs", dist.Normal(gene_patterns, std), obs=data)

        self._last_basis = basis
        self._last_patterns = gene_patterns

        if self.return_basis:
            return basis
        else:
            return gene_patterns

    def get_basis(self, times):
        # noinspection PyTypeChecker
        basis = [trajectory(times) for trajectory in self.trajectory_basis]
        basis = [b / (b.max() + 1e-6) for b in basis]
        basis = torch.cat(basis, dim=1)
        return basis

    def show_basis(self, times):
        # noinspection PyTypeChecker
        basis = np.array(
            [trajectory(times).detach().numpy().reshape(-1) for trajectory in self.trajectory_basis]
        ).T

        data = pd.DataFrame(basis)
        data.plot()
