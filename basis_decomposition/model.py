import pyro
from pyro.nn import PyroModule
import pyro.distributions as dist

import torch
import numpy as np
import pandas as pd


class RealFunction(pyro.nn.PyroModule):
    def __init__(self, *n_hiddens):
        super(RealFunction, self).__init__()
        dimensions = [1, *n_hiddens, 1]
        layers = []
        for in_features, out_features in zip(dimensions, dimensions[1:]):
            layer = pyro.nn.PyroModule[torch.nn.Linear](in_features, out_features)
            layer.weight = pyro.nn.PyroSample(
                dist.Normal(0.0, 1.0 / in_features ** 0.5).expand([out_features, in_features]).to_event(2)
            )
            layer.bias = pyro.nn.PyroSample(
                dist.Normal(0.0, 1.0 / in_features ** 0.5).expand([out_features]).to_event(1)
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
        return x


class TrajectoryModel(torch.nn.Module):
    def __init__(self, K, n_genes, n_conditions=2):
        super().__init__()
        self.K = K
        self.trajectory_basis = pyro.nn.PyroModule[torch.nn.ModuleList](
            [RealFunction(32, 32) for _ in range(self.K)]
        )
        self.n_conditions = n_conditions
        self.n_genes = n_genes
        self.K = K

        self.return_basis = True

    def forward(self, times, data):
        basis = self.get_basis(times)
        # + self.bias

        k_axis = pyro.plate("k", self.K, dim=-1)
        gene_axis = pyro.plate("g", self.n_genes, dim=-2)
        c_axis = pyro.plate("c", self.n_conditions, dim=-3)

        #         with c_axis, gene_axis, k_axis:
        #             betas = pyro.sample(
        #                 "beta",
        #                 dist.Exponential(1)
        #             )
        betas = pyro.sample(
            "beta",
            dist.Exponential(1).expand([self.n_conditions, self.n_genes, self.K]).to_event(3),
        )

        trajectories = torch.einsum("cgk, tk -> cgt", betas, basis)

        t_axis = pyro.plate("t", len(times), dim=-1)

        with c_axis, gene_axis, t_axis:
            pyro.sample("obs", dist.Normal(trajectories, 0.1), obs=data)

        # trajectories = torch.exp(trajectories)
        if self.return_basis:
            return basis
        else:
            return trajectories

    #     def basis_coefficient_l1(self):
    #         return torch.abs(F.softplus(self.weights)).mean() * self.K #+ torch.abs(self.bias).sum()

    #     def basis_regularization(self, times):
    #         return self.get_basis(times).abs().mean()

    def get_basis(self, times):
        basis = [trajectory(times) for trajectory in self.trajectory_basis]
        basis = torch.cat(basis, dim=1)
        return basis

    def show_basis(self, times):
        basis = np.array(
            [trajectory(times).detach().numpy().reshape(-1) for trajectory in self.trajectory_basis]
        ).T

        data = pd.DataFrame(basis)
        data.plot()
