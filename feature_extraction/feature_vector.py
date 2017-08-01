# -*- coding: utf-8 -*-
from nltk.util import ngrams

from grammar.chunker import Chunker
from grammar.pattern_grammar import PatternGrammar
from main import get_dataset

frequent_word_pairs = None
K = 200
import pandas as pd

from spacy.en import English

parser = English()
import os

from collections import Counter


def extract_top_syntactic_grammar_trio():
    top_syntactic_grammar_trio_file = 'top_syntactic_grammar_trio_file.pkl'
    if os.path.isfile(top_syntactic_grammar_trio_file):
        return pd.read_pickle(top_syntactic_grammar_trio_file)

    dataset = get_dataset()
    trio_counter = Counter()
    for data in dataset:
        sentence = data['sentence']
        syntactic_compiled_grammar = PatternGrammar().compile_all_syntactic_grammar()
        # for grammar in syntactic_compiled_grammar

        for index, compiled_grammar in sorted(syntactic_compiled_grammar.items(), key=lambda x: x, reverse=True):
            combos = extract_syntactic_grammar(sentence, grammar=compiled_grammar)
            trio_counter += Counter(combos)

    frequent_trio_counter = sorted(list(dict(trio_counter.most_common(K)).keys()))  # return the actual Counter object
    pd.to_pickle(frequent_trio_counter, top_syntactic_grammar_trio_file)
    return frequent_trio_counter


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
            trigrams = ngrams(pos_tags, 3)
            trigrams_list = [' '.join(trigram) for trigram in trigrams]

    return trigrams_list

# if __name__ == '__main__':
#     df = get_dataset_dataframe()
#     print(get_dataset_dictionary())
