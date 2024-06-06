from argparse import ArgumentParser
from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from dataset_cleanup import _snips_close_field, _fluent_speech_commands


def _decision_tree(df):
    df_train = df[df.split == "train"]
    clf = DecisionTreeClassifier()
    clf.fit(df_train.feat.tolist(), df_train.label)
    df["pred"] = clf.predict(df.feat.tolist())
    return df


def _mlp(df, layer_index, epochs, batch_size=100):
    def _feat(rows):
        return rows.feat.apply(lambda f: f[layer_index]).tolist()

    df_train = df[df.split == "train"]
    df_val = df[df.split == "valid"]

    clf = MLPClassifier(random_state=42, hidden_layer_sizes=(), verbose=0, early_stopping=False)
    best_acc, best_clf = -1, None
    for epoch in range(epochs):
        indices = df_train.index.copy().tolist()
        np.random.shuffle(indices)
        for iter in np.arange(0, len(indices), batch_size):
            batch = df_train.loc[indices[iter:iter+batch_size]]
            clf.partial_fit(_feat(batch), batch.label, classes=df.label.unique())
        val_acc = (clf.predict(_feat(df_val)) == df_val.label).mean()
        if val_acc >= best_acc:
            best_acc = val_acc
            best_clf = deepcopy(clf)

    df["pred"] = best_clf.predict(_feat(df))
    return df


def _relabel(df, split_path):
    if split_path.name == "original_splits":
        return df
    if "snips" in str(split_path):
        new_df = _snips_close_field(split_path.parent, splits=split_path.name)
    else:
        new_df = _fluent_speech_commands(split_path.parent, splits=split_path.name)
    df["key"] = df["key"].astype("string")
    new_df["key"] = new_df["key"].astype("string")
    return df.drop(columns=["label", "split"]).join(new_df[["key", "split", "label"]].set_index("key"), on="key", lsuffix="_r", validate="1:1").dropna()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--df_path", type=Path, help="Path to features")
    parser.add_argument("--split_path", type=Path, help="Path to labels")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train")
    args = parser.parse_args()
    print(args)

    acc_result_path = str(args.df_path).replace(".feat.pkl", f"_{args.split_path.stem}_acc.pkl")
    if not Path(acc_result_path).exists():
        df = _relabel(pd.read_pickle(args.df_path), args.split_path)
        layer_count = df.iloc[0].feat.shape[0]

        accs = []
        for layer in tqdm(range(layer_count)):
            _df = _mlp(df, layer_index=layer, epochs=args.epochs)
            accs.append({
                split: (__df.label == __df.pred).mean()
                for split, __df in _df.groupby("split")
            })
        pickle.dump(accs, open(acc_result_path, "wb"))
    else:
        accs = pickle.load(open(acc_result_path, "rb"))
