import json

import fasttext
import numpy as np
import pandas as pd
from jiwer import wer
from nltk.corpus import wordnet as wn
from scipy.stats import pearsonr, spearmanr
from scipy import spatial
from tqdm import tqdm


def phonetic_dist(x: list[str], y: list[str]):
    ref, hyp = (x, y) if len(x) > len(y) else (y, x)
    return wer(reference=" ".join(ref), hypothesis=" ".join(hyp))


def wordnet_path_sim(w1, w2):
    w1 = wn.synsets(w1)[0]
    w2 = wn.synsets(w2)[0]
    return wn.path_similarity(w1, w2)


def euclidean_dist(f1, f2):
    f1 = np.array(f1)
    f2 = np.array(f2)
    return np.sqrt(np.square(f1-f2).sum())


def dot_sim(f1, f2):
    f1 = np.array(f1)
    f2 = np.array(f2)
    return np.dot(f1, f2)


def cos_sim(f1, f2):
    f1 = np.array(f1)
    f2 = np.array(f2)
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))


def fasttext_sim(w1, w2, model=fasttext.load_model("./datasets/crawl-300d-2M-subword.bin")):
    w1 = model.get_word_vector(w1)
    w2 = model.get_word_vector(w2)
    return spatial.distance.cosine(w1, w2)


def _load_semcor(path):
    semcor = []
    with open(path) as f:
        for line in f.readlines():
            word, scores = line.strip().split("\t")
            scores = json.loads(scores)
            scores["word"] = word
            semcor.append(scores)
    return pd.DataFrame(semcor).replace(np.nan, 0.0).set_index("word")


def semcor_dist(w1, w2, semcor=_load_semcor("./datasets/semcor_noun_verb.supersenses.en")):
    if w1 in semcor.index and w2 in semcor.index:
        f1 = semcor.loc[w1].to_numpy()
        f2 = semcor.loc[w2].to_numpy()
        return euclidean_dist(f1, f2)
    return np.nan


def pearson_corr(d1, d2):
    return pearsonr(d1, d2).statistic


def spearman_corr(d1, d2):
    return spearmanr(d1, d2).statistic


def filter_df(df, speaker, n_sample, seed):
    print(f"Original size: {len(df)}")
    if speaker is not None:
        df = df[df.speaker == speaker]
    if n_sample is not None:
        df = df.sample(n_sample, random_state=seed)
    print(f"Filtered size: {len(df)}")
    return df


def _random_sampler(df, wordmap):
    l_indices = df.index.to_numpy()
    r_indices = l_indices.copy()
    np.random.default_rng(seed=42).shuffle(r_indices)
    for l, r in zip(l_indices, r_indices):
        if l != r:
            yield l, r

def _synonym_sampler(df, wordmap):
    l_indices = df.index.to_numpy()
    for l in l_indices:
        syn = set(wordmap["synonym_map"].get(df.loc[l].text, []))
        for r in df[df.text.isin(syn)].index:
            if l != r:
                yield l, r

def _homophone_sampler(df, wordmap):
    l_indices = df.index.to_numpy()
    for l in l_indices:
        hom = wordmap["homophone_map"].get(df.loc[l].text)
        for r in df[(df.text == hom)].index:
            if l != r:
                yield l, r

def _speaker_sampler(df, wordmap):
    for spk in df.speaker.unique():
        l_indices = df[df.speaker == spk].index.to_numpy()
        r_indices = l_indices.copy()
        np.random.default_rng(seed=42).shuffle(r_indices)
        for l, r in zip(l_indices, r_indices):
            if l != r:
                yield l, r

def _same_word_sampler(df, wordmap):
    for word in df.text.unique():
        l_indices = df[df.text == word].index.to_numpy()
        r_indices = l_indices.copy()
        np.random.default_rng(seed=42).shuffle(r_indices)
        for l, r in zip(l_indices, r_indices):
            if l != r:
                yield l, r


samplers = {
    "random": _random_sampler,
    "synonym": _synonym_sampler,
    "homophone": _homophone_sampler,
    "speaker": _speaker_sampler,
    "same_word": _same_word_sampler,
}
