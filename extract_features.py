import argparse
import functools
import pickle
from pathlib import Path

import librosa
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/wavlm-large", help="Huggingface model name")
    parser.add_argument("--df_path", type=Path, help="Dataset df to inference")
    parser.add_argument("--output_path", type=Path, help="Output df pkl path")
    parser.add_argument("--device", default="cpu", help="Device to infer, cpu or cuda:0 (gpu)")
    return parser.parse_args()


def _get_feat(row, feats):
    index = int((row["min"] + row["max"]) / 2.0 * 16000) // 320
    f = feats[row.audio]
    if index >= len(f):
        index = len(f) - 1
    return f[index]


if __name__ == "__main__":
    args = _get_args()

    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device)

    df = pd.read_csv(args.dataset_csv)
    data = {}
    for path in tqdm(df.path.unique()):
        x, _ = librosa.load(path, sr=16000, mono=True)
        x = processor(raw_speech=[x], sampling_rate=16000, padding=False, return_tensors="pt")
        outputs = model(**{k: t.to(args.device) for k, t in x.items()})
        data[path] = outputs.last_hidden_state.cpu().detach().numpy()[0]

    with open(args.output_path.parent / f"{args.output_path.stem}.raw.pkl", "wb") as f:
        pickle.dump(data, f)

    df["feat"] = df.apply(functools.partial(_get_feat, feats=data), axis=1)
    df.to_pickle(args.output_path)
