import argparse
import json
import gzip
import re
from pathlib import Path

import cmudict
import epitran
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
# from datasets import load_dataset
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


def _wordnet(word, lang="eng"):
    synonyms = [
        syn
        for synsets in wordnet.synsets(word, lang=lang)
        for syn in synsets.lemma_names("eng")
        if syn != word
    ]
    return set(synonyms)


def _librispeech(dataset_path: Path, textgrid_path: Path):
    rows = []
    for split in ("dev-clean", "test-clean"):
        for p in tqdm(textgrid_path.glob(f"{split}/*/*/*.TextGrid")):
            grid = TextGrid(p)
            for word in grid["words"]:
                phones = _cmudict(word.text)
                synonyms = list(_wordnet(word.text))
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


def _multilingual_spoken_words(dataset_path: Path, textgrid_path: Path = None):
    langs = pd.DataFrame([
        {"name": "English", "wordnet": "eng", "MSW": "en", "epitran": "eng-Latn"},
        {"name": "Chinese", "wordnet": "cmn", "MSW": "zh-CN", "epitran": "cmn-Hans"},
        # {"name": "Arabic", "wordnet": "arb", "MSW": "ar"},
        # {"name": "Greek", "wordnet": "ell", "MSW": "el"},
        # {"name": "Persian", "wordnet": "fas", "MSW": "fa"},
        # {"name": "French", "wordnet": "fra", "MSW": "fr"},
        {"name": "Italian", "wordnet": "ita", "MSW": "it", "epitran": "ita-Latn"},
        # {"name": "Catalan", "wordnet": "cat", "MSW": "ca"},
        # {"name": "Basque", "wordnet": "eus", "MSW": "eu"},
        {"name": "Spanish", "wordnet": "spa", "MSW": "es", "epitran": "spa-Latn"},
        {"name": "Indonesian", "wordnet": "ind", "MSW": "id", "epitran": "ind-Latn"},
        {"name": "Polish", "wordnet": "pol", "MSW": "pl", "epitran": "pol-Latn"},
        # {"name": "Portuguese", "wordnet": "por", "MSW": "pt"},
        # {"name": "Slovenian", "wordnet": "slv", "MSW": "sl"},
        {"name": "Swedish", "wordnet": "swe", "MSW": "sv-SE", "epitran": "swe-Latn"},
    ])

    metadata = json.load(gzip.open(dataset_path / "metadata.json.gz"))

    eng_df = pd.read_csv("datasets/msw_english.csv", keep_default_na=False)
    eng_dict = dict(zip(eng_df.word, eng_df.ipa))
    eng_words = set(eng_dict.keys())

    rows = []
    for row in langs.itertuples():
        epi = epitran.Epitran(row.epitran, cedict_file="datasets/cedict_1_0_ts_utf-8_mdbg.txt")
        for word, fnames in tqdm(metadata[row.MSW]["filenames"].items()):
            synonyms = _wordnet(word, lang=row.wordnet) & eng_words
            if len(synonyms) > 0 or row.name == "English":
                for fname in fnames:
                    if row.name == "English":
                        # epitran English is too slow
                        # used cached version in datasets/ directory
                        phones = eng_dict[word]
                        synonyms = []
                    else:
                        phones = epi.transliterate(word)
                    rows.append({
                        "text": word,
                        "path": dataset_path / "audio" / row.MSW / "clips" / word / fname,
                        "phones": list(phones),
                        "synonyms": list(synonyms),
                        "language": row.name,
                        "start": 0.0,
                        "finish": 1.0,
                    })
    return pd.DataFrame(rows)


def _commonvoice_sts(dataset_path: Path, textgrid_path: Path = None):
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
            sf.write(path_b, row["audio_b"], samplerate=16000)
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


def _spoken_sts(dataset_path: Path, textgrid_path: Path = None):
    similarities = json.load(open(dataset_path / "all_gt.json"))

    train_size = len(similarities) // 2
    test_size = len(similarities) - train_size
    splits = np.array(["train"] * train_size + ["test"] * test_size)
    np.random.RandomState(42).shuffle(splits)
    splits = dict(zip(sorted(similarities.keys()), splits))

    rows = []
    for dataset_name in tqdm(("STS12", "STS13", "STS14", "STS15", "STS16", )):
        for subset_path in (dataset_path / dataset_name).glob("*.json"):
            subset_name = subset_path.stem
            for key, (sentence_a, sentence_b) in json.load(open(subset_path)).items():
                for utt_index in range(1, 5):
                    utt_path = dataset_path / "SpokenSTS_wav" / "natural_speech" / dataset_name / subset_name
                    path_a = utt_path / f"{key}_0_human-speaker-{utt_index}.wav"
                    path_b = utt_path / f"{key}_1_human-speaker-{utt_index}.wav"
                    rows.append({
                        "text": sentence_a,
                        "start": 0.0,
                        "finish": librosa.get_duration(path=path_a),
                        "path": path_a,
                        "paired_text": sentence_b,
                        "similarity": similarities[f"{dataset_name}_{key}"],
                        "split": splits[f"{dataset_name}_{key}"],
                    })
                    rows.append({
                        "text": sentence_b,
                        "start": 0.0,
                        "finish": librosa.get_duration(path=path_b),
                        "path": path_b,
                        "paired_text": None,
                        "similarity": None,
                        "split": splits[f"{dataset_name}_{key}"],
                    })
    return pd.DataFrame(rows)


def _fluent_speech_commands(dataset_path: Path, textgrid_path: Path = None, splits: str = "original_splits"):
    dfs = []
    for path in (dataset_path / splits).glob("*.csv"):
        print(path)
        df = pd.read_csv(path, index_col=0)
        split = path.stem.replace("_data", "")
        df = df.rename(columns={"path": "key", "speakerId": "speaker", "transcription": "text"})
        df["split"] = split
        df["label"] = 0
        df["path"] = df["key"].apply(lambda k: dataset_path / k)
        if splits == "original_splits":
            df["start"] = 0.0
            df["finish"] = df["path"].apply(lambda p: librosa.get_duration(path=p))
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    for i, (_, _df) in enumerate(df.groupby(["action", "object", "location"])):
        df.loc[_df.index, "label"] = i
    return df


def _snips_close_field(dataset_path: Path, textgrid_path: Path = None, splits: str = "original_splits"):
    dfs = []
    for path in list((dataset_path / splits).glob("*.csv")) + list((dataset_path / splits).glob("data/*.csv")):
        print(path)
        df = pd.read_csv(path)
        split = path.stem.replace("_data", "")
        df = df.rename(columns={"path": "key", "speakerId": "speaker", "intentLbl": "label", "transcription": "text"})
        df["split"] = split
        df["path"] = df["key"].apply(lambda k: dataset_path / k[3:])
        if splits == "original_splits":
            df["start"] = 0.0
            df["finish"] = df["path"].apply(lambda p: librosa.get_duration(path=p))
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


_SUPPORTED_DATASETS = {
    "librispeech": _librispeech,
    "multilingual_spoken_words": _multilingual_spoken_words,
    "commonvoice_sts": _commonvoice_sts,
    "spoken_sts": _spoken_sts,
    "fluent_speech_commands": _fluent_speech_commands,
    "snips_close_field": _snips_close_field,
}


if __name__ == "__main__":
    args = _get_args()
    parser = _SUPPORTED_DATASETS[args.dataset_type]
    df = parser(dataset_path=args.dataset_path, textgrid_path=args.textgrid_path)
    df.to_pickle(args.output_path)
