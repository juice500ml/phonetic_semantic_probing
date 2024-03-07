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
    parser.add_argument("--homophone_no_filter", type=lambda x: x.lower()=="true", default=False, help="Do not filter for homophone extraction")
    parser.add_argument("--language_uniform", type=lambda x: x.lower()=="true", default=False, help="Sample uniformly per language")
    parser.add_argument("--num_workers", type=int, default=64, help="Number of workers")
    parser.add_argument("--seed", type=int, default=0, help="Number of workers")

    return parser.parse_args()


def _get_synonym_map(words, df, text2phones, threshold=0.4):
    synonym_map = {}
    for index in tqdm(df.reset_index().groupby(["text"])["index"].min()):
        row = df.loc[index]
        synonyms = []
        for s in row.synonyms:
            if threshold < 0:
                synonyms.append(s)
            else:
                if (s in words) and (phonetic_dist(row.phones, text2phones[s]) > threshold):
                    synonyms.append(s)
        synonyms = set(synonyms).intersection(words)
        if len(synonyms) > 0:
            synonym_map[row.text] = synonyms
    return synonym_map


def _get_homophone(df, indices, text2phones, synonym_map, word, threshold=0.4):
    homophones = []
    for index in indices:
        row = df.loc[index]
        if (row.text != word) and \
            (row.text not in synonym_map.get(word, set())) and \
            (word not in synonym_map.get(row.text, set())):
            if 0.0 < phonetic_dist(text2phones[word], row.phones) <= threshold:
                homophones.append(row.text)
    return set(homophones)


def _get_homophone_map(words, synonym_map, df, text2phones, num_workers):
    tqdm_args = dict(max_workers=num_workers, chunksize=len(words) // (num_workers*4))
    indices = df.reset_index().groupby(["text"])["index"].min()

    homophones_list = process_map(
        partial(_get_homophone, df[["text", "phones"]], indices, text2phones, synonym_map),
        words, **tqdm_args)
    return {w: hs for w, hs in zip(words, homophones_list) if len(hs) > 0}


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
    filtered_df = filter_df(df, args.speaker, args.n_sample, args.language_uniform, args.seed)
    words = set(filtered_df.text.unique())
    text2phones = {row.text: tuple(row.phones) for row in filtered_df.itertuples()}

    synonym_map = _get_synonym_map(words, filtered_df, text2phones)
    print(f"Synonym pairs: {sum(len(v) for v in synonym_map.values())}")

    not_filtered_synonym_map = _get_synonym_map(words, filtered_df, text2phones, threshold=-1)
    if args.language_uniform:
        homophone_map = _get_homophone_map(
            set(filtered_df[filtered_df.language == "English"].text.unique()),
            not_filtered_synonym_map,
            df[df.language != "English"] if args.homophone_no_filter else filtered_df[filtered_df.language != "English"],
            text2phones,
            args.num_workers
        )
    else:
        homophone_map = _get_homophone_map(
            words,
            not_filtered_synonym_map,
            df if args.homophone_no_filter else filtered_df,
            text2phones,
            args.num_workers
        )
    print(f"Near-homophone pairs: {sum(len(v) for v in homophone_map.values())}")

    with open(_get_output_path(args), "wb") as f:
        pickle.dump({
            "synonym_map": synonym_map,
            "homophone_map": homophone_map,
        }, f)
