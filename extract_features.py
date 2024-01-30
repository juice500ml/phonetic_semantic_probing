import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, WhisperFeatureExtractor, AutoModel
from tqdm import tqdm

from utils import filter_df


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/wavlm-large", help="Huggingface model name")
    parser.add_argument("--df_path", type=Path, help="Dataset df to inference")
    parser.add_argument("--device", default="cpu", help="Device to infer, cpu or cuda:0 (gpu)")
    parser.add_argument("--speaker", type=str, default=None, help="Speaker id to filter")
    parser.add_argument("--n_sample", type=int, default=None, help="# of random samples")
    parser.add_argument("--pooling", type=str, choices=["center", "mean"], help="Feature pooling method")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for random sampling")
    parser.add_argument("--slice", type=bool, default=False, help="Slice input by word boundary")
    return parser.parse_args()


def _get_index(second, num_frames, model):
    if "whisper" in model:
        index = int(second / 2.0 * 50)
    else:
        index = int(second / 2.0 * 16000) // 320
    if index >= num_frames:
        index = num_frames - 1
    return index


def _get_feat(row, feats, model, pooling, slice):
    if slice:
        start_index = 0
        finish_index = len(feats)
    else:
        start_index = _get_index(
            second=row["start"], num_frames=len(feats), model=model,
        )
        finish_index = _get_index(
            second=row["finish"], num_frames=len(feats), model=model,
        )
    if pooling == "center":
        index = (start_index + finish_index) // 2
        return feats[index]
    elif pooling == "mean":
        return feats[start_index:finish_index+1].mean(0)
    elif pooling == "median_euclidean":
        feats_mean = _get_feat(row, feats, model, "mean")
        dists = ((feats[start_index:finish_index+1] - feats_mean[:, None]) ** 2).sum(1)
        return feats[dists.argmin(1)]
    elif pooling == "median_cosine":
        norm_feats /= (feats ** 2).sum(1, keepdims=True)
        return _get_feat(row, norm_feats, model, "median_euclidean")
    else:
        raise NotImplementedError


def _get_feats(row, feats, model, pooling, slice):
    return np.array([_get_feat(row, f, model, pooling, slice) for f in feats])


def _prepare_input_slice(path, start, finish, model, device):
    x, _ = librosa.load(path, sr=16000, mono=True)
    start = np.clip(int(start * 16000), 0, len(x))
    finish = np.clip(int(finish * 16000), 0, len(x))
    return _preprocess_input(x[start:finish], model, device)


def _prepare_input(path, model, device):
    x, _ = librosa.load(path, sr=16000, mono=True)
    return _preprocess_input(x, model, device)


def _preprocess_input(x, model, device):
    if "whisper" not in model:
        x = processor(raw_speech=[x], sampling_rate=16000, padding=False, return_tensors="pt")
    else:
        # Fixed 30s input for Whisper
        if len(x) > 16000 * 30:
            return None
        x = processor(raw_speech=[x], sampling_rate=16000, return_tensors="pt")
    return {k: t.to(device) for k, t in x.items()}


def _get_output_path(args):
    p = args.df_path
    model = args.model.split("/")[-1]
    speaker = args.speaker if args.speaker is not None else "everyone"
    n_sample = str(args.n_sample) if args.n_sample is not None else "full"
    pool = args.pooling
    seed = args.seed
    slice = args.slice

    fname = f"{p.stem.replace('.df', '')}_model-{model}_slice-{slice}_spk-{speaker}_size-{n_sample}_pool-{pool}_seed-{seed}.feat.pkl"
    return p.parent / fname


def _get_model(model):
    if "whisper" not in model:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model)
        model = AutoModel.from_pretrained(model)
    else:
        processor = WhisperFeatureExtractor.from_pretrained(model)
        model = AutoModel.from_pretrained(model).encoder
    return processor, model


if __name__ == "__main__":
    args = _get_args()
    print(args)

    processor, model = _get_model(args.model)
    model.to(args.device)

    df = pd.read_pickle(args.df_path)
    df = filter_df(df, args.speaker, args.n_sample, args.seed)
    df["feat"] = None

    if args.slice:
        for row in tqdm(df.itertuples()):
            inputs = _prepare_input_slice(row.path, row.start, row.finish, args.model, args.device)
            if inputs is None:
                continue
            outputs = model(**inputs, output_hidden_states=True)
            feats = [h.cpu().detach().numpy()[0] for h in outputs.hidden_states]
            df.at[row.Index, "feat"] = _get_feats(row, feats, args.model, args.pooling, args.slice)

    else:
        for path in tqdm(df.path.unique()):
            inputs = _prepare_input(path, args.model, args.device)
            if inputs is None:
                continue
            outputs = model(**inputs, output_hidden_states=True)
            feats = [h.cpu().detach().numpy()[0] for h in outputs.hidden_states]

            for row in df[df.path == path].itertuples():
                df.at[row.Index, "feat"] = _get_feats(row, feats, args.model, args.pooling, args.slice)

    df.to_pickle(_get_output_path(args))
