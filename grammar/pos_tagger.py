#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nltk
from nltk import PerceptronTagger


class PosTagger:
    def __init__(self, sentence):
        """

        Args:
            sentence:
        """
        self.sentence = sentence
        self.tagger = PosTagger.get_tagger()

    def pos_tag(self):
        """

        Returns:

        """
        tokens = nltk.word_tokenize(self.sentence)
        pos_tagged_tokens = self.tagger.tag(tokens)
        return pos_tagged_tokens

    @staticmethod
    def get_tagger():
        """

        Returns:

        """
        return PerceptronTagger()
