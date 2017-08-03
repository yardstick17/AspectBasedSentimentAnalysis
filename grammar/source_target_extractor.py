#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from collections import defaultdict

import nltk
from nltk.corpus import stopwords

from grammar.chunker import Chunker
from grammar.language_processor import LanguageProcessor
from grammar.pattern_grammar import Target
from grammar.pos_tagger import PosTagger
from grammar.sentiment import Sentiment

STOP_WORDS = set(stopwords.words('english'))
# logging.info("STOP_WORDS SIZE: {}".format(len(STOP_WORDS)))


class SourceTargetExtractor(LanguageProcessor):
    def __init__(self, text):
        """

        :param text: can be a review or raw text
        """
        super().__init__()
        self.text = text
        self.sentences = nltk.sent_tokenize(self.text)
        self.pos_tagged_sentences = [PosTagger(sentence=sentence).pos_tag() for sentence in self.sentences]

    def get_topic_sentiment_score_dict(self, compiled_grammar):
        # self._logger.debug('Getting topic sentiment')
        source_target_mapping = self.get_source_and_target(compiled_grammar)
        source_target_score_mapping = {}
        for source, targets_with_polarity_dict in source_target_mapping.items():
            if targets_with_polarity_dict:
                target_pos_neg_scores = list(
                        map(Sentiment.get_sentiment_with_polarity, set(targets_with_polarity_dict)))
                target_pos_neg_score = self.get_target_pos_neg_scores_mean(target_pos_neg_scores)
                source = self.remove_stop_words(source)
                source_target_score_mapping[source] = target_pos_neg_score
                self._logger.debug(
                        'Source: %s Target %s Target Score: %s',
                        source, str(targets_with_polarity_dict), str(target_pos_neg_score)
                        )
        return source_target_score_mapping

    def remove_stop_words(self, word):
        tokens = word.split()
        new_word = ' '.join([token for token in tokens if token not in STOP_WORDS]).strip()
        if word != new_word:
            logging.info('word: {} , new word: {}'.format(word, new_word))
        return new_word

    def get_source_and_target(self, compiled_grammar):
        subject_to_target_mapping = {}
        chunk_dict = self._get_source_target(compiled_grammar)
        for k, v in chunk_dict.items():
            if subject_to_target_mapping.get(k):
                subject_to_target_mapping[k].extend(v)
            else:
                subject_to_target_mapping.update({k: v})
        return subject_to_target_mapping

    @staticmethod
    def assign_source_and_target(source_set, target_set, polarity, subject_to_target_mapping):
        for subject in source_set:
            # subject = strip_to_root_word(subject)
            if subject:
                for target in target_set:
                    if target not in source_set:
                        subject_to_target_mapping[subject].append(Target(target, polarity))
        return subject_to_target_mapping

    def _get_source_target(self, grammar):
        chunk_dict = {}
        for pos_tagged_sentence in self.pos_tagged_sentences:
            single_chunk_dict = Chunker(grammar).chunk_pos_tagged_sentence(pos_tagged_sentence)
            chunk_dict = self.merge_two_dict(chunk_dict, single_chunk_dict)
        subject_to_target_mapping = defaultdict(list)
        for rule, pos_tagged_chunk_list in chunk_dict.items():
            for pos_tagged_chunk in pos_tagged_chunk_list:
                source_chunk, target_tuple_with_polarity = self.extract_src_target_chunk(rule, pos_tagged_chunk)
                source_set, target_set = self.get_source_target_set(source_chunk, target_tuple_with_polarity)
                self.assign_source_and_target(source_set, target_set, target_tuple_with_polarity.polarity,
                                              subject_to_target_mapping)
        return subject_to_target_mapping
