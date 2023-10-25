import sys
import time

import numpy as np
import pyro
import pyro.infer
import pyro.optim
import torch
from pyro.infer import SVI, Trace_ELBO, Predictive
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from basis_decomposition.model import BasisDecomposition
from basis_decomposition.inference import get_inference_guide, InferenceMode

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def compute_basis_decomposition(
    gene_patterns,
    inference_mode,
    n_basis=5,
    lr=1e-3,
    n_iter=10_000,
    beta_prior=1.0,
    seed=0,
    normalized_mode=True,
    times=None,
):
    pyro.set_rng_seed(seed)
    gene_patterns = torch.FloatTensor(gene_patterns)
    model = BasisDecomposition(
        n_basis,
        n_genes=gene_patterns.shape[1],
        n_conditions=gene_patterns.shape[0],
        beta_prior=beta_prior,
        normalized_mode=normalized_mode,
    )
    guide = get_inference_guide(model, inference_mode)
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    pyro.clear_param_store()
    num_iterations = n_iter
    if times is None:
        times = torch.FloatTensor(np.linspace(-10, 10, gene_patterns.shape[-1]))
    else:
        # TODO: must find a way to ensure that the times are in [-5, 5]-ish
        times = torch.FloatTensor(times)

    gene_patterns_mean = gene_patterns.mean(axis=(0, 2), keepdim=True)
    gene_patterns_raw = gene_patterns
    gene_patterns = gene_patterns_raw / gene_patterns_mean

    pbar = tqdm(range(num_iterations))

    for _ in pbar:
        # calculate the loss and take a gradient step
        loss = svi.step(times, gene_patterns)
        reconstruction = ((model._last_patterns - gene_patterns) ** 2).mean().item()
        reconstruction_rel = reconstruction / (gene_patterns ** 2).mean()

        pbar.set_description(
            "Loss: %.1f - Relative Error: %.2f%%" % (loss, reconstruction_rel * 100)
        )

    model.return_basis = False
    predictive = Predictive(
        model, guide=guide, num_samples=10, return_sites=("beta", "_RETURN", "obs")
    )
    samples = predictive(times, gene_patterns)
    samples["_RETURN"] *= gene_patterns_mean
    gene_scales = model.gene_scales * gene_patterns_mean.squeeze(-1)
    samples = summary(samples)

    return model, guide, times, samples, gene_scales


def main():
    gene_expression, clusters, times = load_simulation_data(sys.argv[1])

    gene_expression = torch.FloatTensor(gene_expression).unsqueeze(0)
    times = torch.FloatTensor(times)

    model = BasisDecomposition(
        10, n_genes=gene_expression.shape[1], n_conditions=gene_expression.shape[0]
    )
    guide = get_inference_guide(model, InferenceMode.GAUSSIAN_BETA_ONLY)

    adam = pyro.optim.Adam({"lr": 1e-3})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    pyro.clear_param_store()
    num_iterations = 100_000

    best_reconstruction_rel = 100
    patience = 200
    acc = 0
    start_time = time.time()

    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(times, gene_expression)
        if j % 100 == 0:
            model.return_basis = False
            predictive = Predictive(
                model, guide=guide, num_samples=10, return_sites=("beta", "_RETURN", "obs")
            )
            samples = predictive(times, gene_expression)
            samples = summary(samples)
            reconstruction = ((samples["_RETURN"]["mean"] - gene_expression) ** 2).mean().item()
            reconstruction_rel = reconstruction / (gene_expression ** 2).mean()
            print(
                "[iteration %04d] loss: %.4f" % (j + 1, loss / len(gene_expression)),
                reconstruction,
                reconstruction_rel,
            )
            best_reconstruction_rel = min(best_reconstruction_rel, reconstruction_rel)
            if reconstruction_rel > best_reconstruction_rel:
                acc += 100
            else:
                acc = 0
            print(acc)
            if acc >= patience:
                print(evaluate(model, guide, clusters, times, gene_expression, max(clusters) + 1))
                print("Total time", time.time() - start_time)
                print("Time/iteration", (time.time() - start_time) / (j + 1))
                print("it/s", (j + 1) / (time.time() - start_time))
                exit()
            if reconstruction_rel < 0.025:
                # print(evaluate(model, guide, clusters, times, gene_expression, max(clusters) + 1))
                exit()
        # if j % 500 == 0:
        #    print(evaluate(model, guide, clusters, times, gene_expression, max(clusters) + 1))

        # plots(model, guide, clusters, times, gene_expression, max(clusters) + 1)


def get_basis(model, guide, gene_patterns, times):
    gene_patterns = torch.FloatTensor(gene_patterns)
    times = torch.FloatTensor(times)
    return_basis_value = model.return_basis
    model.return_basis = True
    predictive = Predictive(
        model, guide=guide, num_samples=2, return_sites=("beta", "_RETURN", "obs")
    )
    samples = predictive(times, gene_patterns)
    samples = summary(samples)
    bases = samples["_RETURN"]["mean"].detach().numpy()
    model.return_basis = return_basis_value
    return bases


def plot_basis(bases, colors=None):
    for i in range(bases.shape[1]):
        plt.plot(
            bases[:, i],
            c=colors[i] if colors is not None else None,
            label="basis %d" % (i + 1),
            linewidth=3,
        )


def plots(model, guide, true_clusters, times, gene_expression, n_clusters):
    model.return_basis = False
    predictive = Predictive(
        model, guide=guide, num_samples=50, return_sites=("beta", "_RETURN", "obs")
    )
    samples = predictive(times, gene_expression)
    samples = summary(samples)

    colors = sns.color_palette(n_colors=10)
    each_cluster = dict.fromkeys(set(true_clusters), 0)
    for i, c in enumerate(true_clusters):
        if c in each_cluster:
            each_cluster[c] += 1
            if each_cluster[c] == 2:
                each_cluster.pop(c)

            plt.plot(samples["_RETURN"]["mean"].squeeze()[i], c=colors[c])
            plt.plot(gene_expression.squeeze()[i], c=colors[c], linestyle="--")
    plt.show()


def evaluate(model, guide, true_clusters, times, gene_expression, n_clusters):
    model.return_basis = False
    predictive = Predictive(model, guide=guide, num_samples=10, return_sites=("beta",))
    samples = predictive(times, gene_expression)
    betas = summary(samples)["beta"]["mean"].squeeze()

    plt.figure(dpi=200)
    sns.heatmap(
        np.exp(
            -((betas[:, None, :] - betas[None, :, :]) ** 2 / ((betas ** 2).mean() * 2)).sum(axis=2)
        ),
        cmap="viridis",
        yticklabels=False,
        xticklabels=False,
        rasterized=True,
    )
    plt.savefig(
        "heatmap.fm.hq.pdf",
        bbox_inches="tight",
    )
    plt.show()
    # do the distance heatmap

    # betas_show = []
    # each_cluster = dict.fromkeys(set(true_clusters), 0)
    # for i, c in enumerate(true_clusters):
    #     if c in each_cluster:
    #         each_cluster[c] += 1
    #         if each_cluster[c] == 3:
    #             each_cluster.pop(c)
    #         betas_show.append(betas[i].detach().numpy())
    #
    # plt.imshow(np.array(betas_show))
    # plt.colorbar()
    # plt.show()

    score_kmeans = []
    for k in range(2, 15):
        kmeans = KMeans(
            n_clusters=k,
        )
        predicted_clusters = kmeans.fit_predict(betas)
        score = silhouette_score(betas, predicted_clusters)
        ari = compute_cluster_score(predicted_clusters, true_clusters)
        score_kmeans.append((k, score, ari))
    print(sorted(score_kmeans, key=lambda x: x[1], reverse=True))

    score_dbscan = []
    for eps in np.linspace(0.1, 1, 10):
        db = DBSCAN(eps=eps)
        predicted_clusters = db.fit_predict(betas)
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        if n_clusters < 2:
            continue
        score = silhouette_score(betas, predicted_clusters)
        ari = compute_cluster_score(predicted_clusters, true_clusters)
        score_dbscan.append((eps, score, ari, n_clusters))
    print(sorted(score_dbscan, key=lambda x: x[1], reverse=True))

    return


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "values": v,
        }
    return site_stats


def compute_cluster_score(predicted_clusters, true_clusters):
    return adjusted_rand_score(true_clusters, predicted_clusters)


def load_simulation_data(path):
    data = np.load(path)
    gene_expression = data["gene_expression"]
    clusters = data["clusters"]
    times = np.arange(gene_expression.shape[1])

    return gene_expression, clusters, times


if __name__ == "__main__":
    pyro.set_rng_seed(0)
    main()
