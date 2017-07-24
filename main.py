# -*- coding: utf-8 -*-
import os
from collections import Counter

import pandas as pd

from dataset.read_dataset import read_json_formatted
from grammar.pattern_grammar import PatternGrammar
from grammar.source_target_extractor import SourceTargetExtractor


def initialize_globals():
    PatternGrammar().compile_all_source_target_grammar()
    PatternGrammar().compile_all_syntactic_grammar()


def get_dataset():
    dataset_filename = 'dataset/annoted_dataset.pkl'
    if not os.path.isfile(dataset_filename):
        annoted_data = read_json_formatted()
        dataset = []
        for row in annoted_data:
            sources = [s.lower() for s in row['target']]
            targets = [s.lower() for s in row['polarity']]
            sentence_meta = {}
            sentence = row['sentence']
            for source, target in zip(sources, targets):
                sentence_meta[source] = target
            dataset.append({'sentence': sentence, 'meta': sentence_meta})
        pd.to_pickle(dataset, dataset_filename)
    else:
        dataset = pd.read_pickle(dataset_filename)
    return dataset


def process():
    initialize_globals()
    annoted_data_dataset = get_dataset()

    syntactic_compiled_grammar = PatternGrammar().compile_all_syntactic_grammar()
    lol_rule_that_extracted_correctly = []
    correct_predictions = 0
    empty_correct_predictions = 0
    index_coverage = Counter()
    for row in annoted_data_dataset:
        sentence = row['sentence'].lower()
        meta = row['meta']
        ste = SourceTargetExtractor(sentence)
        rule_that_extracted_correctly = []
        for index, compiled_grammar in sorted(syntactic_compiled_grammar.items(), key=lambda x: x, reverse=True):
            score_dict = ste.get_topic_sentiment_score_dict(compiled_grammar)
            extracted_meta = {}
            for source, score in score_dict.items():
                if score['PosScore'] < score['NegScore']:
                    extracted_meta[source] = 'negative'
                else:
                    extracted_meta[source] = 'positive'

            if extracted_meta and extracted_meta == meta:
                print(index, 'extracted_meta: ', extracted_meta, ', meta: ', meta)
                correct_predictions += 1
                rule_that_extracted_correctly.append(1)
                index_coverage[index] += 1
            elif extracted_meta == meta:
                empty_correct_predictions += 1
            else:
                rule_that_extracted_correctly.append(0)
        print(rule_that_extracted_correctly)
        lol_rule_that_extracted_correctly.append(rule_that_extracted_correctly)

    print('Correct Predictions:', correct_predictions, 'Empty Correct Predictions :', empty_correct_predictions,
          'Data-set Size :', len(annoted_data_dataset))

    print('Most Efficient Rule: ', list(index_coverage.most_common()))
    print('Rules that at least hit one correct: ', list(index_coverage.keys()))


if __name__ == '__main__':
    process()
