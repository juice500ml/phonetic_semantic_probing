import argparse
import re
from pathlib import Path

import cmudict
import pandas as pd
from nltk.corpus import wordnet
from textgrids import TextGrid
from tqdm import tqdm


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, help="Path to dataset")
    parser.add_argument("--textgrid_path", type=Path, help="Path to TextGrids")
    parser.add_argument("--dataset_type", type=str, choices=["librispeech", ])
    parser.add_argument("--output_path", type=Path, help="Output csv folder")

    return parser.parse_args()


def _cmudict(word, cache=cmudict.dict()):
    if word not in cache:
        return None
    return [
        re.sub(r"\d+", "", p)
        for p in cache[word][0]
    ]


def _wordnet(word):
    synomyms = [
        syn
        for synsets in wordnet.synsets(word)
        for syn in synsets.lemma_names()
        if syn != word
    ]
    return list(set(synomyms))


def _librispeech(dataset_path: Path, textgrid_path: Path):
    rows = []
    for p in tqdm(textgrid_path.glob("train-clean-*/*/*/*.TextGrid")):
        grid = TextGrid(p)
        for word in grid["words"]:
            phones = _cmudict(word.text)
            synonyms = _wordnet(word.text)
            if phones is not None and len(synonyms) > 0:
                rows.append({
                    "text": word.text,
                    "start": word.xmin,
                    "finish": word.xmax,
                    "path": str((dataset_path / p.relative_to(p.parents[3]).with_suffix(".flac")).absolute()),
                    "phones": phones,
                    "synonyms": synonyms,
                    "speaker": p.parents[1].name,
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = _get_args()
    parser = {"librispeech": _librispeech}[args.dataset_type]
    df = parser(dataset_path=args.dataset_path, textgrid_path=args.textgrid_path)
    df.to_pickle(args.output_path)
