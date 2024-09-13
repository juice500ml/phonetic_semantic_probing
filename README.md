# Self-Supervised Speech Representations are More Phonetic than Semantic [![DOI](https://zenodo.org/badge/718434125.svg)](https://zenodo.org/doi/10.5281/zenodo.12741309)
- Accepted to Interspeech 2024
- Paper: https://arxiv.org/abs/2406.08619

![_Interspeech_2024__Poster___Semantic_Probing](https://github.com/user-attachments/assets/e917177f-035a-412e-8edd-9d2a7644ca13)


## Reproducing the experiments
### Environment
- Python 3.9
- Recommended to use conda
- Please check the `requirements.txt` for more details.

### Dataset download
- Please check `datasets/download_datasets.sh` for details.
- It also contains non-dataset stuff.
- For FSC and SNIPS, there is no public URL available. Please follow the instruction of https://github.com/maseEval/mase

### Dataset cleanup
- `dataset_cleanup.py` reformats different datasets into a unified format of `DATASET_NAME.df.pkl`, which is pandas-readable (via `read_pickle` method) dataset dataframe.
- `extract_synonyms_homophones.py` determines the synonyms and near-homophones for each of the word.
  - As it is computationally heavy, we sample LibriSpeech before determining the actual synonyms and near-homophones (i.e., bootstrapping in Sec 2.2 or speaker sampling in Sec 3.5).
  - For MSW, we also balance the language distribution while sampling.
  - For FSC/SNIPS, we do not extract synonyms and near-homophones.
  - The script saves the pickled python dictionary.
  - Ex. `librispeech-dev-clean-test-clean_spk-everyone_size-10000_seed-0.wordmap.pkl` (Sampled 10k words with random seed = 0 for librispeech dev-clean + test-clean).

```sh
# Librispeech
python3 dataset_cleanup.py --dataset_type librispeech \
    --dataset_path datasets/librispeech/LibriSpeech \
    --textgrid_path datasets/librispeech_alignments \
    --output_path tables/librispeech-dev-clean-test-clean.df.pkl

# Librispeech: Bootstrapping
for seed in 0 1 2 3 4; do
    python3 extract_synonyms_homophones.py \
        --df_path tables/librispeech-dev-clean-test-clean.df.pkl \
        --n_sample 10000 --seed $seed --num_workers 32
done

# Librispeech: Speaker sampling
for spk in 5142 2412 6313 1580 2277; do
    python3 extract_synonyms_homophones.py \
        --df_path tables/librispeech-dev-clean-test-clean.df.pkl \
        --speaker $spk --num_workers 32
done


# MSW
python3 dataset_cleanup.py \
    --dataset_path datasets/Multilingual_Spoken_Words \
    --dataset_type multilingual_spoken_words \
    --output_path tables/MSW.pkl

# MSW: Bootstrapping
for seed in 0 1 2 3 4; do
    python3 extract_synonyms_homophones.py \
        --df_path tables/MSW.pkl \
        --n_sample 2000 \
        --language_uniform True \
        --seed $seed
done

# FSC
python3 dataset_cleanup.py --dataset_type fluent_speech_commands \
    --dataset_path datasets/mase/slu_splits/fluent_speech_commands \
    --output_path tables/fluent_speech_commands.df.pkl

# SNIPS
python3 dataset_cleanup.py --dataset_type snips_close_field \
    --dataset_path datasets/mase/slu_splits/snips_close_field \
    --output_path tables/snips_close_field.df.pkl
```


## SSL representation extraction (Librispeech)
- `extract_features.py` reads the dataset dataframe `DATASET_NAME.df.pkl` and inserts the SSL representations to yield the `VARIOUS_SETTINGS.feat.pkl`, which adds the `"feat"` column to the original dataset dataframe.
- There are multiple ways to extract the features, so the settings are written in the file name.
- Ex. `librispeech-dev-clean-test-clean_model-hubert-large-ll60k_slice-True_spk-everyone_size-10000_pool-mean_seed-0.feat.pkl`
- We use the same `filter_df` function for both `extract_features.py` and `extract_synonyms_homophones.py` to filter out the dataset dataframe for LibriSpeech and MSW. Be careful about the randomness. Running on the same environment is recommended.

```sh
model="facebook/hubert-large-ll60k"
# Choose between: microsoft/wavlm-large openai/whisper-large-v3 facebook/wav2vec2-xls-r-300m facebook/hubert-large-ll60k facebook/wav2vec2-base facebook/wav2vec2-large facebook/hubert-base-ls960

pool="mean"
# Default is "mean".
# Consider "center" or "median_cosine" to compare different pooling methods. (Section 3.2)

slice="True"
# Default is "True" (audio slicing).
# Consider "False" for feature slicing. (Section 3.1)

# For standard bootstrapping
for seed in 0 1 2 3 4; do
    python3 extract_features.py \
        --model $model \
        --pooling $pool \
        --slice $slice \
        --n_sample 10000 \
        --seed $seed \
        --device cuda:0 \
        --df_path tables/librispeech-dev-clean-test-clean.df.pkl
done

# For speaker-dependent extraction (Section 3.5)
for spk in 5142 2412 6313 1580 2277; do
    python3 extract_features.py \
        --model hubert-large-ll60k \
        --pooling mean \
        --slice True \
        --speaker $spk \
        --device cuda:0 \
        --df_path tables/librispeech-dev-clean-test-clean.df.pkl
done
```

## SSL representation extraction (Other datasets)
```sh
for model in microsoft/wavlm-large openai/whisper-large-v3 facebook/wav2vec2-xls-r-300m facebook/hubert-large-ll60k facebook/wav2vec2-base facebook/wav2vec2-large facebook/hubert-base-ls960; do
    # MSW: Bootstrapping
    for seed in 0 1 2 3 4; do
        python3 extract_features.py \
            --n_sample 2000 \
            --language_uniform True \
            --model $model \
            --seed $seed \
            --pooling mean --slice True --device cuda:0 \
            --df_path tables/MSW.pkl
    done

    # FSC
    python3 extract_features.py \
        --model $model \
        --pooling mean --slice True --device cuda:0 \
        --df_path tables/fluent_speech_commands.df.pkl

    # SNIPS
    python3 extract_features.py \
        --model $model \
        --pooling mean --slice True --device cuda:0 \
        --df_path tables/snips_close_field.df.pkl
done
```

## Librispeech/MSW word pair distance measurements
- `generate_layerwise_figures_seeds.py` compares the layerwise distances of SSL representations given various types of word pairs.
- It uses `samplers` from `utils.py` for sampling word pairs, i.e., random, synonym, near-homophone, same speaker, and same word. If all speakers are the same, it skips same speaker sampler.
- The distances are cached as `VARIOUS_SETTINGS.dist.pkl` and plotted at `figs/seedwise/VARIOUS_SETTINGS.pdf`.
- Ex. distances: `librispeech-dev-clean-test-clean_model-hubert-large-ll60k_slice-True_spk-everyone_size-10000_pool-mean_seed-0_dist-cos_sim.dist.pkl`
- Ex. figure: `librispeech-dev-clean-test-clean_model-hubert-large-ll60k_slice-True_spk-everyone_size-10000_pool-mean_num_seeds-5_dist-cos_sim_norm-none.pdf`

```sh
dataset="librispeech-dev-clean-test-clean"
# Choose between: librispeech-dev-clean-test-clean MSW

model="hubert-large-ll60k"
# Choose between: wavlm-large wav2vec2-xls-r-300m hubert-large-ll60k wav2vec2-large wav2vec2-base hubert-base-ls960

dist="cos_sim"
# Default is cosine similarity.
# Choose between: cos_sim euclidean_dist dot_sim

pool="mean"
# Default is "mean".
# Consider "center" or "median_cosine" to compare different pooling methods. (Section 3.2)

slice="True"
# Default is "True" (audio slicing).
# Consider "False" for feature slicing. (Section 3.1)

# For standard bootstrapping
python3 generate_layerwise_figures_seeds.py \
    --dataset $dataset \
    --model $model \
    --dist $dist \
    --pooling $pool \
    --slice $slice \
    --size 10000 --seeds 0 1 2 3 4

# For speaker-dependent extraction (Section 3.5)
python3 generate_layerwise_figures_seeds.py \
    --dataset librispeech-dev-clean-test-clean \
    --model $model \
    --dist $dist \
    --pooling $pool \
    --slice $slice \
    --size full --seeds 0 --speakers 5142 2412 6313 1580 2277
```

## FSC/SNIPS intent classification accuracies
- `generate_classifier_figures.py` compares the layerwise accuracies of SSL representations attached with linear probing.
- The accuracies are cached as `VARIOUS_SETTINGS_acc.pkl`.
- Ex. `fluent_speech_commands_model-wavlm-large_slice-True_spk-everyone_size-full_pool-mean_seed-0_challenge_splits_acc.pkl`

```sh
# FSC: Evaluation through all the models
for path in tables/fluent_speech_commands_model*; do
    for split in original_splits challenge_splits; do
        python3 generate_classifier_figures.py \
        --df_path $path \
        --split_path datasets/mase/slu_splits/fluent_speech_commands/$split
    done
done

# SNIPS: Evaluation through all the models
for path in tables/snips_close_field_model*; do
    for split in original_splits challenge_splits; do
        python3 generate_classifier_figures.py \
        --df_path $path \
        --split_path datasets/mase/slu_splits/snips_close_field/$split
    done
done
```

## To reproduce the figures
- Run `draw_figures.ipynb` via `jupyterlab`.
- It contains more refined versions of Librispeech, MSW, FSC, and SNIPS.
- It uses the cached distances and accuracies from `generate_layerwise_figures_seeds.py` and `generate_classifier_figures.py`.
- It also includes the Bag-of-Words + Decision tree baseline for FSC and SNIPS.
