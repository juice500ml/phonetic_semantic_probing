import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from utils import (
    samplers,
    euclidean_dist,
    phonetic_dist,
    wordnet_path_sim,
    semcor_dist,
    spearman_corr,
    fasttext_sim,
)


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["librispeech-train-clean"])
    parser.add_argument("--model", choices=["wavlm-large", "whisper-large-v3"])
    parser.add_argument("--speaker", choices=["everyone", "3072", "8635", "1629"])
    parser.add_argument("--size", choices=["full", "5000"])
    parser.add_argument("--pooling", choices=["center", "mean"])
    parser.add_argument("--layer", choices=["last"])
    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()
    print(args)

    # Load data
    df = pd.read_pickle(f"tables/{args.dataset}_model-{args.model}_spk-{args.speaker}_size-{args.size}_pool-{args.pooling}_layer-{args.layer}.feat.pkl")
    wordmap = pickle.load(open(f"tables/{args.dataset}_spk-{args.speaker}_size-{args.size}.wordmap.pkl", "rb"))

    # Pairwise dists
    dists = {}
    for name, sampler in samplers.items():
        dists[name] = [
            euclidean_dist(df.loc[l].feat, df.loc[r].feat)
            for l, r in tqdm(sampler(df, wordmap))
        ]

    for name, dist in dists.items():
        plt.hist(dist, bins=30, density=True, alpha=0.3, label=f"{name} ({np.mean(dist):.3f})")
    plt.title("Pairwise distance")
    plt.legend()
    p = Path(f"figs/pairwise_dist/{args.dataset}_model-{args.model}_spk-{args.speaker}_size-{args.size}_pool-{args.pooling}_layer-{args.layer}.pdf")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, dist in dists.items():
        sns.kdeplot(data=dist, label=f"{name} ({np.mean(dist):.3f})")
    ax.legend()
    p = Path(f"figs/pairwise_dist_simple/{args.dataset}_model-{args.model}_spk-{args.speaker}_size-{args.size}_pool-{args.pooling}_layer-{args.layer}.pdf")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)

    # Phonetic dists
    phonetic_dists = {}
    for name, sampler in samplers.items():
        phonetic_dists[name] = [
            phonetic_dist(df.loc[l].phones, df.loc[r].phones)
            for l, r in tqdm(sampler(df, wordmap))
        ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for i, name in enumerate(["random", "homophone"]):
        axes[0].scatter(x=dists[name], y=phonetic_dists[name], label=f"{name} ({spearman_corr(dists[name], phonetic_dists[name]):.3f})", alpha=0.5, s=0.1)
        axes[i+1].scatter(x=dists[name], y=phonetic_dists[name], label=f"{name} ({spearman_corr(dists[name], phonetic_dists[name]):.3f})", alpha=0.5, s=0.1, c=f"C{i}")
        axes[i+1].legend()
        # sns.kdeplot(x=dists[name], y=phonetic_dists[name], cmap=["Blues", "Reds"][i], fill=True, label=name, alpha=0.5, ax=ax)ncor
    axes[0].legend()
    axes[0].set_xlabel("Euclidean dist.")
    axes[0].set_ylabel("Levenshtein dist.")
    plt.suptitle("Feature distance (x) vs. Phonetic distance (y)")
    p = Path(f"figs/phonetic_dist/{args.dataset}_model-{args.model}_spk-{args.speaker}_size-{args.size}_pool-{args.pooling}_layer-{args.layer}.pdf")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)

    # Semantic dists
    semantic_dists = {}
    for name, sampler in samplers.items():
        semantic_dists[name] = [
            wordnet_path_sim(df.loc[l].text, df.loc[r].text)
            for l, r in tqdm(sampler(df, wordmap))
        ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for i, name in enumerate(["random", "synonym"]):
        axes[0].scatter(x=dists[name], y=semantic_dists[name], label=f"{name} ({spearman_corr(dists[name], phonetic_dists[name]):.3f})", alpha=0.5, s=0.1)
        axes[i+1].scatter(x=dists[name], y=semantic_dists[name], label=f"{name} ({spearman_corr(dists[name], phonetic_dists[name]):.3f})", alpha=0.5, s=0.1, c=f"C{i}")
        axes[i+1].legend()
    axes[0].legend()
    axes[0].set_xlabel("Euclidean dist.")
    axes[0].set_ylabel("Path sim.")
    plt.suptitle("Feature distance (x) vs. Semantic distance (y)")
    p = Path(f"figs/semantic_dist/{args.dataset}_model-{args.model}_spk-{args.speaker}_size-{args.size}_pool-{args.pooling}_layer-{args.layer}.pdf")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)

    # Fasttext dists
    fasttext_dists = {}
    for name, sampler in samplers.items():
        fasttext_dists[name] = [
            fasttext_sim(df.loc[l].text, df.loc[r].text)
            for l, r in tqdm(sampler(df, wordmap))
        ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for i, name in enumerate(["random", "synonym"]):
        axes[0].scatter(x=dists[name], y=fasttext_dists[name], label=f"{name} ({spearman_corr(dists[name], phonetic_dists[name]):.3f})", alpha=0.5, s=0.1)
        axes[i+1].scatter(x=dists[name], y=fasttext_dists[name], label=f"{name} ({spearman_corr(dists[name], phonetic_dists[name]):.3f})", alpha=0.5, s=0.1, c=f"C{i}")
        axes[i+1].legend()
    axes[0].legend()
    axes[0].set_xlabel("Euclidean dist.")
    axes[0].set_ylabel("Fasttext cos sim.")
    plt.suptitle("Feature distance (x) vs. Fasttext cos. dist. (y)")
    p = Path(f"figs/fasttext_dist/{args.dataset}_model-{args.model}_spk-{args.speaker}_size-{args.size}_pool-{args.pooling}_layer-{args.layer}.pdf")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)

    # semcor dists
    semcor_dists = {}
    for name, sampler in samplers.items():
        semcor_dists[name] = np.array([
            semcor_dist(df.loc[l].text, df.loc[r].text)
            for l, r in tqdm(sampler(df, wordmap))
        ])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for i, name in enumerate(["random", "synonym"]):
        mask = ~np.isnan(semcor_dists[name])
        x = np.array(dists[name])[mask]
        y = semcor_dists[name][mask]
        axes[0].scatter(x=x, y=y, label=f"{name} ({spearman_corr(x, y):.3f})", alpha=0.5, s=0.1)
        axes[i+1].scatter(x=x, y=y, label=f"{name} ({spearman_corr(x, y):.3f})", alpha=0.5, s=0.1, c=f"C{i}")
        axes[i+1].legend()
    axes[0].legend()
    axes[0].set_xlabel("Euclidean dist.")
    axes[0].set_ylabel("semcor Euclidean dist.")
    plt.suptitle("Feature distance (x) vs. semcor Euclidean dist. (y)")
    p = Path(f"figs/semcor_dist/{args.dataset}_model-{args.model}_spk-{args.speaker}_size-{args.size}_pool-{args.pooling}_layer-{args.layer}.pdf")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
