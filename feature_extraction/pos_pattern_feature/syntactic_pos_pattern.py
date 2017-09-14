# -*- coding: utf-8 -*-
from collections import Counter

from nltk.util import ngrams
from spacy.en import English

from grammar.chunker import Chunker
from grammar.pattern_grammar import PatternGrammar
from training.mid_stage_prepare_dataset import get_dataset

K = 1000000

NGRAM_RANGE = [3, 4, 5, 6]
parser = English()

top_syntactic_grammar_list = None


class SyntacticPosPatternFeature:
    DATASET_FILE = 'dataset/annoted_data.json'

    def __init__(self):
        pass

    @staticmethod
    def extract_top_syntactic_pos_pattern_from_corpus():
        """

        :return:
        """
        file = 'dataset/annoted_data.json'
        dataset = get_dataset(file)
        pattern_counter = Counter()
        for data in dataset:
            sentence = data['sentence']
            pattern_counter += SyntacticPosPatternFeature.extract_syntactic_rules_from_sentence(sentence)
        sorted_pattern_counter = sorted(
                list(dict(pattern_counter.most_common(K)).keys()))  # return the actual Counter object
        return sorted_pattern_counter

    @staticmethod
    def extract_syntactic_rules_from_sentence(sentence):
        """

        :param sentence:
        :return:
        """
        trio_counter = Counter()
        syntactic_compiled_grammar = PatternGrammar().compile_all_syntactic_grammar()
        for index, compiled_grammar in sorted(syntactic_compiled_grammar.items(), key=lambda x: x, reverse=True):
            combos = SyntacticPosPatternFeature.extract_syntactic_grammar(sentence, grammar=compiled_grammar)
            trio_counter += Counter(combos)
        return trio_counter

    @staticmethod
    def extract_syntactic_grammar(sentence, grammar):
        """

        :param sentence:
        :param grammar:
        :return:
        """
        chunk_dict = Chunker(grammar).chunk_sentence(sentence)
        trigrams_list = []
        for key, pos_tagged_sentences in chunk_dict.items():
            pos_tags = [token[1] for pos_tagged_sentence in pos_tagged_sentences for token in pos_tagged_sentence]
            if len(pos_tags) >= 2:
                for ngram in NGRAM_RANGE:
                    trigrams = ngrams(pos_tags, ngram)
                    trigrams_list.extend([' '.join(trigram).strip() for trigram in trigrams])
        return trigrams_list

    @staticmethod
    def get_top_syntactic_grammar_pos_pattern():
        """

        :return:
        """
        global top_syntactic_grammar_list
        if top_syntactic_grammar_list is None:
            top_syntactic_grammar_list = SyntacticPosPatternFeature.extract_top_syntactic_pos_pattern_from_corpus()
        return top_syntactic_grammar_list
