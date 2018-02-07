#!/usr/bin/env bash
if [[ $1 = 3 ]]; then
  python -m nltk.downloader punkt
  python -m nltk.downloader wordnet
  python -m nltk.downloader sentiwordnet
  python -m nltk.downloader vader_lexicon
  python -m nltk.downloader stopwords
  python -m nltk.downloader averaged_perceptron_tagger
else
  echo "Usage: ./setup.sh <int: python version>"
fi
