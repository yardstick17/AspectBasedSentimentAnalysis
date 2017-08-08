# -*- coding: utf-8 -*-
from feature_extraction.pre_feature_vector_stage import extract_top_syntactic_grammar_trio
from feature_extraction.pre_feature_vector_stage import extract_trio_syntactic_rules

top_syntactic_grammar_list = None


def get_top_syntactic_grammar_list():
    """

    :return:
    """
    global top_syntactic_grammar_list
    if top_syntactic_grammar_list is None:
        top_syntactic_grammar_list = extract_top_syntactic_grammar_trio()
    return top_syntactic_grammar_list


def get_syntactic_grammar_feature(sentence_text):
    """

    :param sentence_text:
    :return:
    """
    trigrams_list = extract_trio_syntactic_rules(sentence_text)
    top_syntactic_grammar_list = get_top_syntactic_grammar_list()
    X = [1 if j in trigrams_list else 0 for i, j in enumerate(top_syntactic_grammar_list)]
    return X
