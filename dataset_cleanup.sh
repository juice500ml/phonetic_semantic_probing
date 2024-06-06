# python3 dataset_cleanup.py --dataset_type librispeech \
#     --dataset_path /scratch/bbjs/shared/corpora/Librispeech/LibriSpeech \
#     --textgrid_path /scratch/bbjs/kchoi1/semantic_probing/datasets/librispeech_alignments \
#     --output_path librispeech-train-clean.df.pkl


python3 dataset_cleanup.py --dataset_type commonvoice_sts \
    --dataset_path  /scratch/bbjs/kchoi1/semantic_probing/datasets/commonvoice_sts \
    --output_path tables/commonvoice_sts.df.pkl
