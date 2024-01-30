python3 dataset_cleanup.py --dataset_type librispeech \
    --dataset_path /scratch/bbjs/shared/corpora/Librispeech/LibriSpeech \
    --textgrid_path /scratch/bbjs/kchoi1/semantic_probing/datasets/librispeech_alignments \
    --output_path librispeech-train-clean.df.pkl

python3 extract_features.py --df_path /scratch/bbjs/kchoi1/semantic_probing/librispeech-train-clean.df.pkl --output_path /scratch/bbjs/kchoi1/semantic_probing/feats.df.pkl --device 0