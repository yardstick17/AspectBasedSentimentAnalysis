# -*- coding: utf-8 -*-
import os
from collections import Counter

import pandas as pd
from nltk.util import ngrams
from spacy.en import English

from grammar.chunker import Chunker
from grammar.pattern_grammar import PatternGrammar
from main import get_dataset

K = 1200
NGRAM_COUNT = 3

parser = English()


def extract_top_syntactic_grammar_trio():
    top_syntactic_grammar_trio_file = 'top_syntactic_grammar_trio_file.pkl'
    if os.path.isfile(top_syntactic_grammar_trio_file):
        return pd.read_pickle(top_syntactic_grammar_trio_file)

    dataset = get_dataset()
    trio_counter = Counter()
    for data in dataset:
        sentence = data['sentence']
        trio_counter += extract_trio_syntactic_rules(sentence)

    frequent_trio_counter = sorted(list(dict(trio_counter.most_common(K)).keys()))  # return the actual Counter object
    pd.to_pickle(frequent_trio_counter, top_syntactic_grammar_trio_file)
    return frequent_trio_counter


def extract_trio_syntactic_rules(sentence):
    trio_counter = Counter()
    syntactic_compiled_grammar = PatternGrammar().compile_all_syntactic_grammar()
    for index, compiled_grammar in sorted(syntactic_compiled_grammar.items(), key=lambda x: x, reverse=True):
        combos = extract_syntactic_grammar(sentence, grammar=compiled_grammar)
        trio_counter += Counter(combos)
    return trio_counter


# TODO : introduce dependency relation later
"""
def extract_dependency_relations(sentence):

    parsedEx = parser(sentence)
    for token in parsedEx:
        print(token.orth_, token.dep_, token.head.orth_)
"""


def extract_syntactic_grammar(sentence, grammar):
    chunk_dict = Chunker(grammar).chunk_sentence(sentence)
    trigrams_list = []
    for key, pos_tagged_sentences in chunk_dict.items():
        pos_tags = [token[1] for pos_tagged_sentence in pos_tagged_sentences for token in pos_tagged_sentence]
        if len(pos_tags) > 2:
            trigrams = ngrams(pos_tags, NGRAM_COUNT)
            trigrams_list = [' '.join(trigram) for trigram in trigrams]

    return trigrams_list
