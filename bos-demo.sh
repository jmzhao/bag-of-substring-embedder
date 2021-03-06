#!/usr/bin/env bash
echo "CAUTION: This will download Google word2vec vectors (3.4 GB)."
read -p "Press enter to continue"

DATADIR=./datasets
RESULTSDIR=./results/bos/demo

mkdir -p "${RESULTSDIR}"
mkdir -p "${DATADIR}"

if [ ! -f "${DATADIR}/word2vec/GoogleNews-vectors-negative300.bin" ]
then
  URL="https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
  mkdir "${DATADIR}/word2vec"
  wget -O "${DATADIR}/word2vec/GoogleNews-vectors-negative300.bin.gz" ${URL}
  echo "Unzipping..."
  gunzip "${DATADIR}/word2vec/GoogleNews-vectors-negative300.bin.gz"
fi
if [ ! -f "${DATADIR}/word2vec/processed.txt" ]
then
  python data-process.py --input "${DATADIR}/word2vec/GoogleNews-vectors-negative300.bin" --output "${DATADIR}/word2vec/processed.txt"
fi

if [ ! -f "${DATADIR}/rw/rw.txt" ]
then
  wget -c 'https://nlp.stanford.edu/~lmthang/morphoNLM/rw.zip' -P "${DATADIR}"
  unzip "${DATADIR}/rw.zip" -d "${DATADIR}"
fi
if [ ! -f "${DATADIR}/rw/queries.txt" ]
then
  cut -f 1,2 "${DATADIR}/rw/rw.txt" | awk '{print tolower($0)}' | tr '\t' '\n' > "${DATADIR}/rw/queries.txt"
fi

python bos-train.py --target "${DATADIR}/word2vec/processed.txt" --save "${RESULTSDIR}" --no-timestamp --epochs 20
python bos-pred.py --queries "${DATADIR}/rw/queries.txt" --save "${RESULTSDIR}/rw_vectors.txt" --model "${RESULTSDIR}/model.bos"
python ./fastText/eval.py --data "${DATADIR}/rw/rw.txt" --model "${RESULTSDIR}/rw_vectors.txt"
