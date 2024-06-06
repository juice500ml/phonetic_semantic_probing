# Librispeech
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz

# Textgrids for Librispeech
wget https://zenodo.org/record/2619474/files/librispeech_alignments.zip
mkdir librispeech_alignments
unzip librispeech_alignments.zip -d librispeech_alignments

# Fasttext model
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
unzip crawl-300d-2M-subword.zip

# SemCor dataset
wget https://raw.githubusercontent.com/ytsvetko/qvec/master/oracles/semcor_noun_verb.supersenses.en

# MSW
wget https://storage.googleapis.com/public-datasets-mswc/mswc.tar.gz
tar -xzf mswc.tar.gz

# MSW/Epitran-related
wget https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz

# Splits for SNIPS/FSC
git clone https://github.com/maseEval/mase.git
echo "Please check the above repository to download SNIPS and FSC."
