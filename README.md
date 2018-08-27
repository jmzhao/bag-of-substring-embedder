# Bag-of-Substring Word Embedder
This is the code used in Generalizing Word Embeddings using Bag of Subwords (EMNLP 2018).

## Dependencies
- Python 3.5
- gensim 3.1 for loading word2vec vectors.
- tqdm 4.15 for progress reporting.
- fastText for word similarity evaluation script.
- Mimick for sentense-level bidirectional LSTM model.

The last two are included as submodules.
Initialize the submodules using
```
git submodule init
git submodule update
```

## Contents
- `bos-demo.sh`: a script showcasing the usage.
- `bos-pred.py`: generate vectors for word queries.
- `bos-train.py`: train a BoS model.
- `bos.py`: BoS model definition.
- `data-processing.py`: prepare target word vectors.

## Usage
To start a demo run, simply `./bos-demo.sh`.

To preform word similarity evluation, refer to `fastText/eval.py`.

To perform POS and morphosyntactic tagging evaluation, refer to `Mimick/model.py`.
