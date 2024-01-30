import argparse
import re
from pathlib import Path

import cmudict
import soundfile as sf
import pandas as pd
from datasets import load_dataset
from nltk.corpus import wordnet
from textgrids import TextGrid
from tqdm import tqdm


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, help="Path to dataset")
    parser.add_argument("--textgrid_path", type=Path, help="Path to TextGrids")
    parser.add_argument("--dataset_type", type=str, choices=list(_SUPPORTED_DATASETS.keys()))
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


def _commonvoice_sts(dataset_path: Path, textgrid_path: Path):
    rows = []
    ds = load_dataset("charsiu/Common_voice_sentence_similarity", cache_dir=dataset_path / "cache")
    for split in ("dev", "test"):
        split_name = "train" if split == "dev" else "test"
        for row in tqdm(ds[split]):
            path_a = dataset_path / f"{Path(row['path_a']).stem}.wav"
            sf.write(path_a, row["audio_a"], samplerate=16000)
            rows.append({
                "text": row["sentence_a"],
                "start": 0.0,
                "finish": len(row["audio_a"]) / 16000,
                "path": path_a,
                "paired_text": row["sentence_b"],
                "similarity": row["similarity"],
                "split": split_name,
            })

            path_b = dataset_path / f"{Path(row['path_b']).stem}.wav"
            sf.write(path_b, row["audio_a"], samplerate=16000)
            rows.append({
                "text": row["sentence_b"],
                "start": 0.0,
                "finish": len(row["audio_b"]) / 16000,
                "path": path_b,
                "paired_text": None,
                "similarity": None,
                "split": split_name,
            })
    return pd.DataFrame(rows)


_SUPPORTED_DATASETS = {
        "librispeech": _librispeech,
        "commonvoice_sts": _commonvoice_sts,
}


if __name__ == "__main__":
    args = _get_args()
    parser = _SUPPORTED_DATASETS[args.dataset_type]
    df = parser(dataset_path=args.dataset_path, textgrid_path=args.textgrid_path)
    df.to_pickle(args.output_path)
