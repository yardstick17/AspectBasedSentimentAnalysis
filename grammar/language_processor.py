# -*- coding: utf-8 -*-
from collections import defaultdict
from collections import namedtuple

import numpy as np
from nltk.sentiment.vader import NEGATE

from grammar.chunker import Chunker
from grammar.sentiment import NEGATIVE_SENTIMENT_SCORE
from grammar.sentiment import POSITIVE_SENTIMENT_SCORE

Target = namedtuple('Target', ['word', 'polarity'])

NEGATE_SET = set(NEGATE) | {"n't", 'never'}


class LanguageProcessor:
    POSITIVE_POLARITY = 'positive'
    NEGATIVE_POLARITY = 'negative'

    def __init__(self):
        import logging
        self._logger = logging.getLogger()

    @staticmethod
    def merge_two_dict(dict_x, dict_y):
        """
        :param dict_x: {'a': [3, 4], 'b': [6]}
        :param dict_y: {'c': [3], 'a': [1, 2]}
        :return: {'c': [3], 'a': [3, 4, 1, 2], 'b': [6]}
        """
        dict_z = dict_x.copy()  # Never modify input param , or take inplace as param for explicit use case
        for key, value in dict_y.items():
            if dict_z.get(key):
                dict_z[key].extend(value)
            else:
                dict_z[key] = value
        return dict_z

    @staticmethod
    def get_source_target_set(source_chunk, target_chunk_with_polarity: Target):
        """
        :param source_chunk: list of source chunk , extracted using
        :param target_chunk_with_polarity:
        :return:
        """
        target_chunk = target_chunk_with_polarity.word
        source_set, target_set = set(), set()

        target_chunk = [tgt for tgt in target_chunk if tgt not in source_chunk]
        for src in source_chunk:
            src_pos_tagged_part = src[0]
            np_phrase_pos_tagged_list = Chunker.get_chunk(src_pos_tagged_part, 'NN_all')
            # np_phrase_pos_tagged_part = np_phrase_pos_tagged_list[0]
            for np_phrase_pos_tagged_part in np_phrase_pos_tagged_list:
                for single_np_phrase in np_phrase_pos_tagged_part:
                    source_word = ' '.join([i[0] for i in single_np_phrase]).strip()
                    source_set.add(source_word)

        for tgt in target_chunk:
            tgt_pos_tagged_part = tgt[0]
            sentiment_phrase_pos_tagged_list = Chunker.get_chunk(tgt_pos_tagged_part, 'JJ_NN_RB_VB')
            # sentiment_phrase_pos_tagged_part = sentiment_phrase_pos_tagged_list[0]
            for sentiment_phrase_pos_tagged_part in sentiment_phrase_pos_tagged_list:
                for single_sentiment_phrase in sentiment_phrase_pos_tagged_part:
                    target_word = ' '.join([i[0] for i in single_sentiment_phrase]).strip()
                    target_set.add(target_word)
        return source_set, target_set

    @staticmethod
    def extract_src_target_chunk(key, pos_tagged_chunk: Target):
        source, target = [], []
        if key in ['JJ_DESCRIBING_NN_V4']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'NP_before_VB')
            target = Chunker.get_chunk(pos_tagged_chunk, 'JJ_AFTER_VB')
            if not target:
                target = Chunker.get_chunk(pos_tagged_chunk, 'NN_JJ_desc')

            if not source or not target:
                source, target_tuple_with_polarity = LanguageProcessor.extract_src_target_chunk('VBG_DESCRIBING_NN_V3',
                                                                                                pos_tagged_chunk)
                return source, target_tuple_with_polarity
        elif key in ['VBG_RB_DESRIBING_NN', 'VBN_DESCRING_THE_FOLLOWING_NOUN', 'VBG_DESRIBING_NN']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'NP_After_VB_i')
            target = Chunker.get_chunk(pos_tagged_chunk, 'VB_JJ_RB_desc')
        elif key in ['JJ_VBG_RB_DESRIBING_NN', 'JJ_VBG_RB_DESRIBING_NN_2']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'NP_before_VB')
            target = Chunker.get_chunk(pos_tagged_chunk, 'NN_JJ_desc')
            target.extend(Chunker.get_chunk(pos_tagged_chunk, 'RB_AFTER_VB'))
            target.extend(Chunker.get_chunk(pos_tagged_chunk, 'VB_JJ_RB_desc'))
            if not source:
                source = Chunker.get_chunk(pos_tagged_chunk, 'NN_all')

            if not source or not target:
                source, target_tuple_with_polarity = LanguageProcessor.extract_src_target_chunk('VBG_DESCRIBING_NN_V3',
                                                                                                pos_tagged_chunk)
                return source, target_tuple_with_polarity

        elif key in ['VBG_DESCRIBING_NN_V3', 'VBG_NN_DESCRIBING_NN']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'NP_After_VB')
            target = Chunker.get_chunk(pos_tagged_chunk, 'VB_all')
        elif key in ['VBG_DESRIBING_NN_V2']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'NP_After_VB')
            target = Chunker.get_chunk(pos_tagged_chunk, 'VB_JJ_RB_desc')
        elif key in ['VBG_DESCRIBIN_NN_V4', 'VB_DESCRBING_NN']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'NP_After_VB')
            target = Chunker.get_chunk(pos_tagged_chunk, 'VB_JJ_RB_desc')
        elif key in ['RB_BEFORE_NN']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'NN_all')
            target = Chunker.get_chunk(pos_tagged_chunk, 'RB_all')
        elif key in ['JJ_BEFORE_NN', 'NN_JJ']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'NN_all')
            target = Chunker.get_chunk(pos_tagged_chunk, 'JJ_all')
        elif key in ['JJ_IN_NN']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'NE_grammar')
            target = Chunker.get_chunk(pos_tagged_chunk, 'JJ_any_IN')
        elif key in ['JJ_TO_NN_VB']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'TO_NN')
            target = Chunker.get_chunk(pos_tagged_chunk, 'JJ_multi')
        elif key in ['NN_MD_VB']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'NE_grammar')
            target = Chunker.get_chunk(pos_tagged_chunk, 'VB_desc')
        elif key in ['VBN_IN_PRP_NN', 'VB_PRP_NNS']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'NP_After_VB')
            target = Chunker.get_chunk(pos_tagged_chunk, 'VB_desc')
        elif key in ['PR_VB_JJ_JJ']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'JJ_multi')
            target = Chunker.get_chunk(pos_tagged_chunk, 'VB_desc')
        elif key in ['JJ_BEFORE_NN_V3', 'I_JJ_NN']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'NN_only')
            target = Chunker.get_chunk(pos_tagged_chunk, 'JJ_multi')
        elif key in ['NN_IN_DT_NN']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'NN_IN')
            target = Chunker.get_chunk(pos_tagged_chunk, 'DT_NN')
        elif key in ['NN_VB_DT_JJ_NN']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'JJ_NN_end')
            target = Chunker.get_chunk(pos_tagged_chunk, 'NN_beg')
        elif key in ['NN_Phrase']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'NN_FW_only')
            target = Chunker.get_chunk(pos_tagged_chunk, 'NN_beg')
        elif key in ['NN_desc_NN']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'NN_beg')
            target = Chunker.get_chunk(pos_tagged_chunk, 'NP_After_VB_must')
        elif key in ['NN_DT_NN', 'NN_desc_NN_reverse', 'NN_IN_DT_NN_reverse']:
            source = Chunker.get_chunk(pos_tagged_chunk, 'DT_NN')
            target = Chunker.get_chunk(pos_tagged_chunk, 'NN_beg')

        polarity = LanguageProcessor.get_polarity(pos_tagged_chunk)
        target_tuple_with_polarity = Target(target, polarity)
        return source, target_tuple_with_polarity

    @staticmethod
    def get_polarity(pos_tagged_chunk: Target):
        negation_word = NEGATE_SET & {pos_tuple[0] for pos_tuple in pos_tagged_chunk}
        return LanguageProcessor.POSITIVE_POLARITY if not negation_word else LanguageProcessor.NEGATIVE_POLARITY

    @staticmethod
    def get_target_pos_neg_scores_mean(target_pos_neg_scores):
        """
        :param target_pos_neg_scores:  [{'NegScore': 0.21875, 'PosScore': 0.375},
                                        {'NegScore': 0.0625, 'PosScore': 0.0},
                                        {'NegScore': 0.21875, 'PosScore': 0.375}]
        :return: {'NegScore': 0.16666, 'PosScore': 0.25}
        """
        pos_scores_mean = np.mean(list(map(lambda x: x[POSITIVE_SENTIMENT_SCORE], target_pos_neg_scores)))
        neg_scores_mean = np.mean(list(map(lambda x: x[NEGATIVE_SENTIMENT_SCORE], target_pos_neg_scores)))
        return {POSITIVE_SENTIMENT_SCORE: pos_scores_mean,
                NEGATIVE_SENTIMENT_SCORE: neg_scores_mean}

    @staticmethod
    def reject_general_english_word(subject_to_target_mapping):
        source_target_mapping_new = defaultdict(list)
        for source, list_of_targets_with_polarity in subject_to_target_mapping.items():
            for target_with_polarity in list_of_targets_with_polarity:
                word = target_with_polarity.word
                if word:
                    source_target_mapping_new[source].append(Target(word=word, polarity=target_with_polarity.polarity))

        return source_target_mapping_new
