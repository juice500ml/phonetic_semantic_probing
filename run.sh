#!/bin/bash

# python3 dataset_cleanup.py --dataset_type librispeech \
#     --dataset_path /scratch/bbjs/shared/corpora/Librispeech/LibriSpeech \
#     --textgrid_path /scratch/bbjs/kchoi1/semantic_probing/datasets/librispeech_alignments \
#     --output_path librispeech-train-clean.df.pkl

# python3 extract_features.py --df_path /scratch/bbjs/kchoi1/semantic_probing/librispeech-train-clean.df.pkl --output_path /scratch/bbjs/kchoi1/semantic_probing/feats.df.pkl --device 0

# FULL EXP
# # for model in microsoft/wavlm-large openai/whisper-large-v3 facebook/wav2vec2-xls-r-300m facebook/hubert-large-ll60k facebook/hubert-base-ls960 facebook/wav2vec2-base facebook/wav2vec2-large; do
# for model in facebook/hubert-large-ll60k ; do
#     for pool in mean; do
#         for slice in True; do
#             # for seed in 0 1 2 3 4; do
#             #     sbatch --account=bbjs-delta-gpu --gres=gpu:1 -p gpuA40x4-interactive -t 1:00:00 --mem 60000M \
#             #         extract_features.sh extract_features.py --model $model --pooling $pool --slice $slice --n_sample 10000 --seed $seed --device cuda:0 --df_path /scratch/bbjs/kchoi1/semantic_probing/tables/librispeech-train-clean.df.pkl
#             # done
#             for spk in 3072 8635 1629 3185 2294; do
#                 sbatch --account=bbjs-delta-gpu --gres=gpu:1 -p gpuA40x4-interactive -t 1:00:00 --mem 60000M \
#                     extract_features.sh extract_features.py --model $model --pooling $pool --slice $slice --speaker $spk --device cuda:0 --df_path /scratch/bbjs/kchoi1/semantic_probing/tables/librispeech-train-clean.df.pkl
#             done
#         done
#     done
# done

# EXP 1
# for model in openai/whisper-large-v3; do
# # for model in microsoft/wavlm-large openai/whisper-large-v3 facebook/wav2vec2-xls-r-300m facebook/hubert-large-ll60k; do
#     # for pool in center mean median_euclidean median_cosine; do
#     for pool in center mean; do
#         for slice in True False; do
#             for seed in 0 1 2 3 4; do
#                 # sbatch --account=bbjs-delta-gpu --gres=gpu:1 -p gpuA40x4-interactive -t 1:00:00 --mem 60000M \
#                 sbatch --account=bbjs-delta-gpu --gres=gpu:1 -t 1:00:00 --mem 20000M \
#                     extract_features.sh extract_features.py --model $model --pooling $pool --slice $slice --n_sample 10000 --seed $seed --device cuda:0 --df_path /scratch/bbjs/kchoi1/semantic_probing/tables/librispeech-train-clean.df.pkl
#             done
#         done
#     done
# done

# # EXP 3
# for model in openai/whisper-large-v3 facebook/wav2vec2-xls-r-300m facebook/hubert-large-ll60k; do
#     for pool in center mean median_euclidean median_cosine; do
#         for slice in True; do
#             for seed in 0; do
#                 sbatch --account=bbjs-delta-gpu --gres=gpu:1 -p gpuA40x4-interactive -t 1:00:00 --mem 60000M \
#                     extract_features.sh extract_features.py --model $model --pooling $pool --slice $slice --n_sample 10000 --seed $seed --device cuda:0 --df_path /scratch/bbjs/kchoi1/semantic_probing/tables/librispeech-train-clean.df.pkl
#             done
#             # for spk in 3072 8635 1629 3185 2294; do
#             #     sbatch --account=bbjs-delta-gpu --gres=gpu:1 -p gpuA40x4-interactive -t 1:00:00 --mem 60000M \
#             #         extract_features.sh extract_features.py --model $model --pooling $pool --slice $slice --speaker $spk --device cuda:0 --df_path /scratch/bbjs/kchoi1/semantic_probing/tables/librispeech-train-clean.df.pkl
#             # done
#         done
#     done
# done

# # EXP 4
# for model in microsoft/wavlm-large; do
#     for pool in center; do
#         for slice in False; do
#             for seed in 0; do
#                 sbatch --account=bbjs-delta-gpu --gres=gpu:1 -p gpuA40x4-interactive -t 1:00:00 --mem 60000M \
#                     extract_features.sh extract_features.py --model $model --pooling $pool --slice $slice --n_sample 10000 --seed $seed --device cuda:0 --df_path /scratch/bbjs/kchoi1/semantic_probing/tables/librispeech-train-clean.df.pkl
#             done
#         done
#     done
# done

# # EXP 5
# for model in microsoft/wavlm-large; do
#     for pool in center; do
#         for slice in True; do
#             for spk in 3072 8635 1629 3185 2294; do
#                 sbatch --account=bbjs-delta-gpu --gres=gpu:1 -p gpuA40x4-interactive -t 1:00:00 --mem 60000M \
#                     extract_features.sh extract_features.py --model $model --pooling $pool --slice $slice --speaker $spk --device cuda:0 --df_path /scratch/bbjs/kchoi1/semantic_probing/tables/librispeech-train-clean.df.pkl
#             done
#         done
#     done
# done

# for seed in 1 2 3 4; do
#     sbatch --account=bbjs-delta-cpu -p cpu --cpus-per-task=32 -t 1:00:00 \
#         extract_features.sh extract_synonyms_homophones.py --df_path tables/librispeech-train-clean.df.pkl --n_sample 10000 --seed $seed --df_path /scratch/bbjs/kchoi1/semantic_probing/tables/librispeech-train-clean.df.pkl --num_workers 32
# # done
# for spk in 3072 8635 1629 3185 2294; do
#     sbatch --account=bbjs-delta-cpu -p cpu --cpus-per-task=32 -t 1:00:00 \
#         extract_features.sh extract_synonyms_homophones.py --df_path tables/librispeech-train-clean.df.pkl --speaker $spk --df_path /scratch/bbjs/kchoi1/semantic_probing/tables/librispeech-train-clean.df.pkl --num_workers 32
# done



# for model in wavlm-large whisper-large-v3 wav2vec2-xls-r-300m hubert-large-ll60k; do
#     for pool in center mean; do
#         for slice in True False; do
#             for dist in euclidean_dist cos_sim dot_sim; do
#                 for seed in 0 1 2 3 4; do
#                     sbatch --account=bbjs-delta-cpu -p cpu --cpus-per-task=2 -t 1:00:00 -d afterany:2918990 \
#                         extract_features.sh generate_layerwise_figures.py --model $model --pooling $pool --slice $slice --size 5000 --seed $seed --dist $dist
#                 done
#                 for spk in 3072 8635 1629 3185 2294; do
#                     sbatch --account=bbjs-delta-cpu -p cpu --cpus-per-task=2 -t 1:00:00 -d afterany:2918990 \
#                         extract_features.sh generate_layerwise_figures.py --model $model --pooling $pool --slice $slice --speaker $spk --dist $dist
#                 done
#             done
#         done
#     done
# done


# for model in wavlm-large wav2vec2-xls-r-300m hubert-large-ll60k; do
#     for pool in center; do
#         for slice in True; do
#             for dist in euclidean_dist; do
#                 for seed in 0; do
#                     sbatch --account=bbjs-delta-cpu -p cpu --cpus-per-task=2 -t 1:00:00 \
#                         extract_features.sh generate_layerwise_figures.py --model $model --pooling $pool --slice $slice --size 10000 --seed $seed --dist $dist
#                 done
#             done
#         done
#     done
# done




# # EXP 2
# for model in wavlm-large; do
#     for pool in center; do
#         for slice in True; do
#             for dist in euclidean_dist dot_sim; do
#                 for seed in 0; do
#                     sbatch --account=bbjs-delta-cpu -p cpu --cpus-per-task=2 -t 1:00:00 \
#                         extract_features.sh generate_layerwise_figures.py --model $model --pooling $pool --slice $slice --size 10000 --seed $seed --dist $dist
#                 done
#             done
#         done
#     done
# done

# # EXP 3
# for model in wav2vec2-xls-r-300m hubert-large-ll60k whisper-large-v3; do
#     for pool in center; do
#         for slice in True; do
#             for dist in cos_sim; do
#                 for seed in 0; do
#                     sbatch --account=bbjs-delta-cpu -p cpu --cpus-per-task=2 -t 1:00:00 \
#                         extract_features.sh generate_layerwise_figures.py --model $model --pooling $pool --slice $slice --size 10000 --seed $seed --dist $dist
#                 done
#             done
#         done
#     done
# done

# # EXP 4
# for model in wavlm-large; do
#     for pool in center mean; do
#         for slice in True False; do
#             for dist in euclidean_dist; do
#                 for seed in 0; do
#                     sbatch --account=bbjs-delta-cpu -p cpu --cpus-per-task=2 -t 1:00:00 \
#                         extract_features.sh generate_layerwise_figures.py --model $model --pooling $pool --slice $slice --size 10000 --seed $seed --dist $dist
#                 done
#             done
#         done
#     done
# done

# # EXP 5
# for model in wavlm-large; do
#     for pool in center; do
#         for slice in True; do
#             for dist in euclidean_dist; do
#                 for speaker in 3072 8635 1629 3185 2294; do
#                     sbatch --account=bbjs-delta-cpu -p cpu --cpus-per-task=2 -t 1:00:00 \
#                         extract_features.sh generate_layerwise_figures.py --model $model --pooling $pool --slice $slice --speaker $speaker --dist $dist
#                 done
#             done
#         done
#     done
# done




# # ARCHITECTURE
# # for model in wavlm-large whisper-large-v3 wav2vec2-xls-r-300m hubert-large-ll60k hubert-base-ls960 wav2vec2-base wav2vec2-large; do
# for model in whisper-large-v3; do
#     for pool in mean; do
#         for slice in True; do
#             for dist in cos_sim; do
#                 sbatch --account=bbjs-delta-cpu -p cpu --cpus-per-task=8 --mem 8000M -t 2:00:00 \
#                     extract_features.sh generate_layerwise_figures_seeds.py --model $model --pooling $pool --slice $slice --size 10000 --num_seeds 5 --dist $dist
#             done
#         done
#     done
# done

# # SLICING
# for model in hubert-large-ll60k whisper-large-v3; do
#     for pool in mean; do
#         for slice in False; do
#             for dist in cos_sim; do
#                 sbatch --account=bbjs-delta-cpu -p cpu --cpus-per-task=4 --mem 8000M -t 1:00:00 \
#                     extract_features.sh generate_layerwise_figures_seeds.py --model $model --pooling $pool --slice $slice --size 10000 --num_seeds 5 --dist $dist
#             done
#         done
#     done
# done


# # POOLING
# for model in hubert-large-ll60k; do
#     for pool in center median_cosine; do
#         for slice in True; do
#             for dist in cos_sim; do
#                 sbatch --account=bbjs-delta-cpu -p cpu --cpus-per-task=4 --mem 8000M -t 1:00:00 \
#                     extract_features.sh generate_layerwise_figures_seeds.py --model $model --pooling $pool --slice $slice --size 10000 --num_seeds 5 --dist $dist
#             done
#         done
#     done
# done


# # SPEAKER
# for model in hubert-large-ll60k; do
#     for pool in mean; do
#         for slice in True; do
#             for dist in cos_sim; do
#                 sbatch --account=bbjs-delta-cpu -p cpu --cpus-per-task=8 --mem 8000M -t 2:00:00 \
#                     extract_features.sh generate_layerwise_figures_seeds.py --model $model --pooling $pool --slice $slice --size full --num_seeds 1 --speakers 3072 8635 1629 3185 2294 --dist $dist
#             done
#         done
#     done
# done



# envs/bin/python3 dataset_cleanup.py --dataset_type commonvoice_sts \
#     --dataset_path /scratch/bbjs/kchoi1/semantic_probing/datasets/commonvoice_sts \
#     --output_path /scratch/bbjs/kchoi1/semantic_probing/tables/commonvoice_sts.df.pkl

# python3 extract_features.py \
#     --df_path /scratch/bbjs/kchoi1/semantic_probing/commonvoice_sts.df.pkl --output_path /scratch/bbjs/kchoi1/semantic_probing/feats.df.pkl --device 0


# FULL EXP
for model in facebook/hubert-large-ll60k microsoft/wavlm-large openai/whisper-large-v3 facebook/wav2vec2-xls-r-300m facebook/hubert-base-ls960 facebook/wav2vec2-base facebook/wav2vec2-large; do
    for pool in mean median_cosine center; do
        sbatch --account=bbjs-delta-gpu --gres=gpu:1 -p gpuA40x4-interactive -t 1:00:00 --mem 60000M \
            extract_features.sh extract_features.py \
            --model $model --pooling $pool --slice True --device cuda:0 --df_path /scratch/bbjs/kchoi1/semantic_probing/tables/fluent_speech_commands.df.pkl
    done
done
