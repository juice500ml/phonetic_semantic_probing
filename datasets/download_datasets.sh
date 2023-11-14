wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
tar -xzf train-clean-360.tar.gz
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz

wget https://zenodo.org/record/2619474/files/librispeech_alignments.zip
mkdir librispeech_alignments
unzip librispeech_alignments.zip -d librispeech_alignments
