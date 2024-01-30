import argparse
import pickle
import multiprocessing as mp
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from utils import filter_df, phonetic_dist


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", type=Path, help="Path to dataset")
    parser.add_argument("--speaker", type=str, default=None, help="Speaker id to filter")
    parser.add_argument("--n_sample", type=int, default=None, help="# of random samples")
    parser.add_argument("--homophone_no_filter", type=bool, default=False, help="Do not filter for homophone extraction")
    parser.add_argument("--num_workers", type=int, default=64, help="Number of workers")
    parser.add_argument("--seed", type=int, default=0, help="Number of workers")

    return parser.parse_args()



def _get_synonym_map(words, df):
    synonym_map = {}
    for index in tqdm(df.reset_index().groupby(["text"])["index"].min()):
        row = df.loc[index]
        synonyms = set([s for s in row.synonyms if s != row.text])
        if len(synonyms - words) > 0:
            synonym_map[row.text] = synonyms
    return synonym_map


def _get_homophone(df, words, indices, word):
    phones = df[df.text == word].iloc[0].phones
    dist = 2.0
    homophone = None

    for index in indices:
        row = df.loc[index]
        if row.text not in synonym_map[word] and \
            row.text != word and \
            not (row.text in synonym_map and word not in synonym_map[row.text]):
            new_dist = phonetic_dist(phones, row.phones)
            if 0.0 <= new_dist < dist:
                dist = new_dist
                homophone = row.text
    return homophone


def _get_homophone_map(synonym_map, df, num_workers):
    tqdm_args = dict(max_workers=num_workers, chunksize=len(synonym_map) // (num_workers*4))
    words = list(synonym_map.keys())
    indices = df.reset_index().groupby(["text"])["index"].min()
    homophones = process_map(
        partial(_get_homophone, df[["text", "phones"]], words, indices),
        words, **tqdm_args)
    return dict(zip(words, homophones))


def _get_output_path(args):
    p = args.df_path
    speaker = args.speaker if args.speaker is not None else "everyone"
    n_sample = str(args.n_sample) if args.n_sample is not None else "full"
    seed = args.seed

    fname = f"{p.stem.replace('.df', '')}_spk-{speaker}_size-{n_sample}_seed-{seed}.wordmap.pkl"
    return p.parent / fname


if __name__ == "__main__":
    args = _get_args()
    print(args)

    df = pd.read_pickle(args.df_path)
    filtered_df = filter_df(df, args.speaker, args.n_sample, args.seed)
    words = set(filtered_df.text.unique())

    synonym_map = _get_synonym_map(words, filtered_df)
    homophone_map = _get_homophone_map(synonym_map, df if args.homophone_no_filter else filtered_df, args.num_workers)

    with open(_get_output_path(args), "wb") as f:
        pickle.dump({
            "synonym_map": synonym_map,
            "homophone_map": homophone_map,
        }, f)
