# -*- coding: utf-8 -*-
import logging
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

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


def get_polarity(score):
    return 'negative' if score['PosScore'] < score['NegScore'] else 'positive'


def process():
    initialize_globals()
    annoted_data_dataset = get_dataset()

    syntactic_compiled_grammar = PatternGrammar().compile_all_syntactic_grammar()
    correct_predictions = 0
    empty_correct_predictions = 0
    non_empty_miss_case = 0
    index_coverage = Counter()
    lol_mean_match = []
    label = []
    predicted_label = []

    for row in tqdm(annoted_data_dataset):
        sentence = row['sentence'].lower()
        meta = row['meta']
        ste = SourceTargetExtractor(sentence)
        expected_meta_form = set(sorted(meta.items()))
        max_match = 0
        max_tmp_label = []
        max_tmp_predicted_label = []
        final_rule = -1
        for index, compiled_grammar in sorted(syntactic_compiled_grammar.items(), key=lambda x: x, reverse=True):
            score_dict = ste.get_topic_sentiment_score_dict(compiled_grammar)

            extracted_meta_form = get_extracted_meta_information(score_dict)
            all_data = extracted_meta_form | expected_meta_form
            intersection = extracted_meta_form & expected_meta_form
            tmp_label, tmp_predicted_label = build_expected_and_extracted_label_data(expected_meta_form,
                                                                                     extracted_meta_form, intersection)

            final_rule, max_match, max_tmp_label, max_tmp_predicted_label = update_rule_with_extracted_label_with_max_score(
                    all_data, final_rule, index, intersection, max_match, max_tmp_label, max_tmp_predicted_label,
                    tmp_label,
                    tmp_predicted_label)
        if not max_tmp_label:
            """
            case: when there is no subject in the sentence (null in the data-set)
            """
            max_tmp_label.append(0)
            max_tmp_predicted_label.append(0)

        label.extend(max_tmp_label)
        index_coverage[final_rule] += 1
        predicted_label.extend(max_tmp_predicted_label)
        lol_mean_match.append(max_match)

    logging.info(np.mean(lol_mean_match))
    logging.info(
            'Correct Predictions: {}, Empty Correct Predictions : {}, Non empty_miss_case: {}, Data-set Size: {} '.format(
                    correct_predictions,
                    empty_correct_predictions, non_empty_miss_case,
                    len(annoted_data_dataset)))

    logging.info('Most Efficient Rule: %s', list(index_coverage.most_common()))
    logging.info('Rules that at least hit one correct: %s', list(index_coverage.keys()))
    logging.info('\n{}'.format(classification_report(label, predicted_label)))


def update_rule_with_extracted_label_with_max_score(all_data, final_rule, index, intersection, max_match, max_tmp_label,
                                                    max_tmp_predicted_label, tmp_label, tmp_predicted_label):
    if len(all_data):
        match_percent = len(intersection) / len(all_data)
        if max_match < match_percent:  # to avoid update on the null subject cases
            max_match = match_percent
            max_tmp_predicted_label = tmp_predicted_label
            max_tmp_label = tmp_label
            final_rule = index
    return final_rule, max_match, max_tmp_label, max_tmp_predicted_label


def build_expected_and_extracted_label_data(expected_meta_form, extracted_meta_form, intersection):
    tmp_predicted_label = []
    tmp_label = []
    for _ in range(len(intersection)):
        tmp_label.append(1)
        tmp_predicted_label.append(1)
    for _ in range(len(expected_meta_form - extracted_meta_form)):
        tmp_label.append(0)
        tmp_predicted_label.append(1)
    for _ in range(len(extracted_meta_form - expected_meta_form)):
        tmp_label.append(1)
        tmp_predicted_label.append(0)
    return tmp_label, tmp_predicted_label


def get_extracted_meta_information(score_dict):
    extracted_meta = {}
    for source, score in score_dict.items():
        if score['PosScore'] < score['NegScore']:
            extracted_meta[source] = 'negative'
        else:
            extracted_meta[source] = 'positive'
    extracted_meta_form = set(sorted(extracted_meta.items()))
    return extracted_meta_form


if __name__ == '__main__':
    logging.basicConfig(format='[%(name)s] [%(asctime)s] %(levelname)s : %(message)s', level=logging.INFO)
    process()
