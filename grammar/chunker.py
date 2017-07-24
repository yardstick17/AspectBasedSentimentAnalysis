#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nltk

from grammar.pattern_grammar import PatternGrammar
from grammar.pos_tagger import PosTagger


class Chunker:
    def __init__(self, grammar: nltk.RegexpParser):
        self.grammar = grammar

    def chunk_sentence(self, sentence: str):
        pos_tagged_sentence = PosTagger(sentence).pos_tag()
        return self.chunk_pos_tagged_sentence(pos_tagged_sentence)

    def chunk_pos_tagged_sentence(self, pos_tagged_sentence):
        chunked_tree = self.grammar.parse(pos_tagged_sentence)
        chunk_dict = self.extract_rule_and_chunk(chunked_tree)
        return chunk_dict

    def extract_rule_and_chunk(self, chunked_tree: nltk.Tree) -> dict:
        def recursively_get_pos_only(tree, collector_list=None, depth_limit=100):
            if collector_list is None:
                collector_list = []
            if depth_limit <= 0:
                return collector_list
            for subtree in tree:
                if isinstance(subtree, nltk.Tree):
                    recursively_get_pos_only(subtree, collector_list, depth_limit - 1)
                else:
                    collector_list.append(subtree)
            return collector_list

        def get_pos_tagged_and_append_to_chunk_dict(chunk_dict, subtrees):  # params can be removed now
            pos_tagged = recursively_get_pos_only(subtrees)
            chunk_dict[subtrees.label()].append(pos_tagged)

        chunk_dict = nltk.defaultdict(list)
        for subtrees in chunked_tree:
            if isinstance(subtrees, nltk.Tree):
                get_pos_tagged_and_append_to_chunk_dict(chunk_dict, subtrees)
                for sub in subtrees:
                    if isinstance(sub, nltk.Tree):
                        get_pos_tagged_and_append_to_chunk_dict(chunk_dict, sub)
        return chunk_dict

    @staticmethod
    def get_chunk(pos_tagged_sentence, src_target_grammar_key: str) -> list:
        compile_grammar = PatternGrammar().get_source_target_compiled_grammar(clause=src_target_grammar_key)
        return Chunker.apply_grammar_on_pos_tagged_chunk(compile_grammar, pos_tagged_sentence)

    @staticmethod
    def apply_grammar_on_pos_tagged_chunk(compile_grammar, pos_tagged_sentence):
        chunk_dict = Chunker(compile_grammar).chunk_pos_tagged_sentence(pos_tagged_sentence)
        return list(chunk_dict.values()) if chunk_dict else []
