# -*- coding: utf-8 -*-
import functools

import numpy as np
import pandas as pd
from nltk.corpus import sentiwordnet
from nltk.stem import PorterStemmer

SENTINET_PREFIX = 9

ROOT = ''
NEGATIVE_WORDS_TXT = ROOT + 'dataset/sentiment_words_text_files/negative_words.txt'
POSITIVE_WORDS_TXT = ROOT + 'dataset/sentiment_words_text_files/positive_words.txt'
NEUTRAL_MODIFIERS_TXT = ROOT + 'dataset/sentiment_words_text_files/neutral_modifiers.txt'

POSITIVE_SENTIMENT_SCORE = 'PosScore'
NEGATIVE_SENTIMENT_SCORE = 'NegScore'
stemmer = PorterStemmer()

neutral_modifiers = None
positive_modifiers = None
negative_modifiers = None
_POSITIVE_POLARITY = 'positive'
_NEGATIVE_POLARITY = 'negative'


class Sentiment:
    @staticmethod
    @functools.lru_cache(maxsize=16384)
    def get_sentiment_for_word(target: str):
        positive_list_score, negative_list_score = [], []
        adj_list = target.split()  # Handle word like "must try" as "must", "try"
        for adj in adj_list:
            pos, neg = Sentiment._find_sentiment_score_for(adj)
            if pos or neg:
                positive_list_score.append(pos)
                negative_list_score.append(neg)
        positive = np.mean(positive_list_score) if positive_list_score else 0
        negative = np.mean(negative_list_score) if negative_list_score else 0
        return {
            POSITIVE_SENTIMENT_SCORE: positive,
            NEGATIVE_SENTIMENT_SCORE: negative
        }

    @staticmethod
    def get_sentiment_with_polarity(target_polarity):
        sentiment_score = Sentiment.get_sentiment_for_word(target_polarity.word)
        if target_polarity.polarity == 'negative':
            reverse_sentiment_score = {
                POSITIVE_SENTIMENT_SCORE: sentiment_score[NEGATIVE_SENTIMENT_SCORE],
                NEGATIVE_SENTIMENT_SCORE: sentiment_score[POSITIVE_SENTIMENT_SCORE]
            }
            sentiment_score = reverse_sentiment_score
        return sentiment_score

    @staticmethod
    def _find_sentiment_score_for(word):
        pos_matched_word = pd.Series(list(sentiwordnet.senti_synsets(word)))
        neg_matched_word = pd.Series(list(sentiwordnet.senti_synsets(word)))
        pos = pos_matched_word.apply(lambda x: x.pos_score()).mean() if len(pos_matched_word) else 0
        neg = neg_matched_word.apply(lambda x: x.neg_score()).mean() if len(neg_matched_word) else 0
        return pos, neg

    @staticmethod
    def get_neutral_modifiers():
        global neutral_modifiers
        if not neutral_modifiers:
            with open(NEUTRAL_MODIFIERS_TXT, 'r') as txt_file:
                neutral_modifiers = {w for word in txt_file
                                     for w in word.lower().strip()}
                # neutral_modifiers.update({token for word in neutral_modifiers for token in word.split()})
        neutral_modifiers -= Sentiment.get_positive_modifiers()
        neutral_modifiers -= Sentiment.get_negative_modifiers()
        return neutral_modifiers

    @staticmethod
    def get_positive_modifiers():
        global positive_modifiers
        if not positive_modifiers:
            with open(POSITIVE_WORDS_TXT, 'r') as txt_file:
                positive_modifiers = {word.lower().strip() for word in txt_file}
        return positive_modifiers

    @staticmethod
    def get_negative_modifiers():
        global negative_modifiers
        if not negative_modifiers:
            with open(NEGATIVE_WORDS_TXT, 'r') as txt_file:
                negative_modifiers = {word.lower().strip() for word in txt_file}
        return negative_modifiers

    @staticmethod
    def neutral_words(word, stemmed_adj):
        return (word.strip().lower() in Sentiment.get_neutral_modifiers() or
                stemmed_adj.lower() in Sentiment.get_neutral_modifiers())

    @staticmethod
    def positive_words(word, stemmed_adj):
        return (word.strip().lower() in Sentiment.get_positive_modifiers() or
                stemmed_adj.lower() in Sentiment.get_positive_modifiers())

    @staticmethod
    def negative_words(word, stemmed_adj):
        return (word.strip().lower() in Sentiment.get_negative_modifiers() or
                stemmed_adj.lower() in Sentiment.get_negative_modifiers())
