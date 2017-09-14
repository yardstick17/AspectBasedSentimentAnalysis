# -*- coding: utf-8 -*-
from feature_extraction.pos_pattern_feature.syntactic_pos_pattern import SyntacticPosPatternFeature


def get_syntactic_grammar_feature_vector(sentence_text):
    """

    :param sentence_text:
    :return:
    """
    pos_tagger_pattern = SyntacticPosPatternFeature.extract_syntactic_rules_from_sentence(sentence_text)
    top_syntactic_grammar_list = SyntacticPosPatternFeature.get_top_syntactic_grammar_pos_pattern()
    X = [1 if j in pos_tagger_pattern else 0 for i, j in enumerate(top_syntactic_grammar_list)]
    return X
