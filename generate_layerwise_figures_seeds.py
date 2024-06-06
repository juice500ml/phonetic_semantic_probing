import argparse
import pickle
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pandas as pd
from tqdm import tqdm

from utils import (
    samplers,
    euclidean_dist,
    cos_sim,
    dot_sim,
)

def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="librispeech-train-clean")
    parser.add_argument("--model")
    parser.add_argument("--slice")
    parser.add_argument("--speakers", default=["everyone"], nargs="+")
    parser.add_argument("--size", default="full")
    parser.add_argument("--pooling")
    parser.add_argument("--dist")
    parser.add_argument("--seeds", type=int, default=[0, 1, 2, 3, 4], nargs="+")
    return parser.parse_args()


def mean_confidence_interval(data, confidence=0.95):
    # Obtained from https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


if __name__ == "__main__":
    args = _get_args()
    print(args)
    dist_func = {"euclidean_dist": euclidean_dist, "cos_sim": cos_sim, "dot_sim": dot_sim}[args.dist]

    seedwise_dists = []
    for seed, speaker in tqdm(product(args.seeds, args.speakers)):
        dists_path = Path(f"tables/{args.dataset}_model-{args.model}_slice-{args.slice}_spk-{speaker}_size-{args.size}_pool-{args.pooling}_seed-{seed}_dist-{args.dist}.dist.pkl")
        if dists_path.exists():
            dists = pickle.load(open(dists_path, "rb"))
        else:
            # Load data
            df = pd.read_pickle(f"tables/{args.dataset}_model-{args.model}_slice-{args.slice}_spk-{speaker}_size-{args.size}_pool-{args.pooling}_seed-{seed}.feat.pkl")
            wordmap = pickle.load(open(f"tables/{args.dataset}_spk-{speaker}_size-{args.size}_seed-{seed}.wordmap.pkl", "rb"))

            if ("speaker" not in df.keys()) or (len(args.speakers) >= 2):
                if "speaker" in samplers.keys():
                    del samplers["speaker"]

            dists = defaultdict(list)
            layer_count = len(df.iloc[0].feat)
            for layer in tqdm(range(layer_count)):
                for name, sampler in samplers.items():
                    accumulator = defaultdict(list)
                    for l, r in tqdm(sampler(df, wordmap)):
                        accumulator[(df.loc[l].text, df.loc[r].text)].append(cos_sim(df.loc[l].feat[layer], df.loc[r].feat[layer]))
                    dists[name].append(np.array([np.array(v).mean() for v in accumulator.values()]))
            pickle.dump(dists, open(dists_path, "wb"))

        _agg_dists = {
            k: [mean_confidence_interval(v)[0] for v in vs]
            for k, vs in dists.items()
        }
        seedwise_dists.append(_agg_dists)

    agg_dists = {}
    for key in seedwise_dists[0].keys():
        agg_dists[key] = []
        for layer in range(len(seedwise_dists[0]["random"])):
            vs = [seedwise_dists[i][key][layer] for i in range(len(seedwise_dists))]
            agg_dists[key].append(mean_confidence_interval(vs))

    for normalizer in ("none", "subtract"):
        plt.figure()
        for key, tuples in agg_dists.items():
            value = np.array([t[0] for t in tuples])
            bound = np.array([t[1] for t in tuples])
            if normalizer == "subtract":
                value -= np.array([t[0] for t in agg_dists["random"]])
            plt.plot(value, "o-", label=key)
            plt.fill_between(np.arange(len(value)), value-bound, value+bound, alpha=0.1)
        plt.legend()

        title = f"{args.dataset}_model-{args.model}_slice-{args.slice}_spkcnt-{len(args.speakers)}_size-{args.size}_pool-{args.pooling}_num_seeds-{len(args.seeds)}_dist-{args.dist}_norm-{normalizer}"
        plt.title(title.replace("_pool", "\npool"))

        p = Path(f"figs/seedwise/{title}.pdf")
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p)
