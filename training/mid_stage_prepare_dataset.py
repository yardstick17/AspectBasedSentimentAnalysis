# -*- coding: utf-8 -*-
import logging
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

from dataset.read_dataset import read_json_formatted
from grammar.pattern_grammar import PatternGrammar
from grammar.source_target_extractor import SourceTargetExtractor

LABEL_LIST_PKL = 'label_list.pkl'
MID_TRAINING_DATASET = 'mid_training_data.csv'
grammar_label = None


def initialize_globals():
    PatternGrammar().compile_all_source_target_grammar()
    PatternGrammar().compile_all_syntactic_grammar()


def get_dataset(dataset_filename=None):
    annoted_data = read_json_formatted(dataset_filename)
    dataset = []
    for row in annoted_data:
        sources = [s.lower() for s in row['target']]
        targets = [s.lower() for s in row['polarity']]
        sentence_meta = {}
        sentence = row['sentence']
        for source, target in zip(sources, targets):
            sentence_meta[source] = target
        dataset.append({'sentence': sentence, 'meta': sentence_meta})

    return dataset


def get_polarity(score):
    return 'negative' if score['PosScore'] < score['NegScore'] else 'positive'


def get_syntactic_rules_in_list():
    global grammar_label
    if grammar_label is None:
        grammar_label = pd.read_pickle(LABEL_LIST_PKL)
    return grammar_label


def extract_mid_stage_label_dataset(dataset_filename):
    logging.info('Dataset: {}'.format(dataset_filename))
    initialize_globals()
    annoted_data_dataset = get_dataset(dataset_filename)

    syntactic_compiled_grammar = PatternGrammar().compile_all_syntactic_grammar()
    index_coverage = Counter()
    lol_mean_match = []
    label = []
    predicted_label = []
    mid_training_data = []
    total_aspects = 0
    for row in tqdm(annoted_data_dataset):
        logging.debug('========================================================================================')
        sentence = row['sentence']
        logging.debug('sentence: {}'.format(sentence))

        meta = row['meta']
        meta = {key: value for key, value in meta.items() if key != 'null'}
        expected_meta_form = set(sorted(meta.items()))
        total_aspects += len(expected_meta_form)
        logging.debug('EXPECTED_META_FORM: %s', expected_meta_form)

        ste = SourceTargetExtractor(sentence)
        max_match_percent = 0
        max_tmp_label = []
        max_tmp_predicted_label = []
        final_rule = -1
        for index, compiled_grammar in sorted(syntactic_compiled_grammar.items(), key=lambda x: x, reverse=True):
            score_dict = ste.get_topic_sentiment_score_dict(compiled_grammar)

            extracted_meta = {}
            for source, score in score_dict.items():
                source = source.lower()
                if score['PosScore'] < score['NegScore']:
                    extracted_meta[source] = 'negative'
                else:
                    extracted_meta[source] = 'positive'

            extracted_meta_form = set(sorted(extracted_meta.items()))
            all_data = extracted_meta_form | expected_meta_form
            intersection = extracted_meta_form & expected_meta_form
            tmp_predicted_label = []
            tmp_label = []
            for _ in range(len(intersection)):
                tmp_label.append(1)
                tmp_predicted_label.append(1)

            for _ in range(len(expected_meta_form - extracted_meta_form)):
                tmp_label.append(1)
                tmp_predicted_label.append(0)

            for _ in range(len(extracted_meta_form - expected_meta_form)):
                tmp_label.append(0)
                tmp_predicted_label.append(1)

            if len(all_data):
                match_percent = len(intersection) / len(all_data)
                if max_match_percent <= match_percent:  # to avoid update on the null subject cases
                    logging.debug('------------------------------------------------')
                    logging.debug('extracted_meta_form: %s', extracted_meta_form)
                    logging.debug('expected_meta_form: %s', expected_meta_form)
                    logging.debug('intersection: %s', intersection)
                    logging.debug('------------------------------------------------')
                    max_match_percent = match_percent
                    max_tmp_predicted_label = tmp_predicted_label
                    max_tmp_label = tmp_label
                    final_rule = index
        if len(max_tmp_label) == 0 and len(meta) == 0:
            """
            case: when there is no subject in the sentence (null in the data-set)
            """
            max_tmp_label.append(0)
            max_tmp_predicted_label.append(0)

        logging.debug('***************************************************************')
        logging.debug('expected_label- {}'.format(max_tmp_label))
        logging.debug('extracted_label- {}'.format(max_tmp_predicted_label))
        logging.debug('***************************************************************')
        label.extend(max_tmp_label)
        index_coverage[final_rule] += 1
        predicted_label.extend(max_tmp_predicted_label)
        lol_mean_match.append(max_match_percent)
        mid_training_data.append([sentence, final_rule])

    df = pd.DataFrame(mid_training_data, columns=['sentence', 'label'])

    df.to_csv(MID_TRAINING_DATASET)
    log_stats_of_pre_training_stage(annoted_data_dataset, index_coverage, label, lol_mean_match, predicted_label,
                                    total_aspects)

    global grammar_label
    grammar_label = list(index_coverage.keys())
    pd.to_pickle(list(index_coverage.keys()), LABEL_LIST_PKL)
    return df


def log_stats_of_pre_training_stage(annoted_data_dataset, index_coverage, label, lol_mean_match, predicted_label,
                                    total_aspects):
    logging.info('================================================================')
    logging.info('Data-set Size: {} '.format(len(annoted_data_dataset)))
    logging.info('Total_aspects Size: {} '.format(total_aspects))
    logging.info('Most Efficient Rule: %s', list(index_coverage.most_common()))
    logging.info('Rules that at least hit one correct: %s', list(index_coverage.keys()))
    logging.info('mean of percent extracted targets - {}', np.mean(lol_mean_match))
    logging.info('\n{}'.format(classification_report(label, predicted_label)))
    logging.info('================================================================')


if __name__ == '__main__':
    logging.basicConfig(format='[%(name)s] [%(asctime)s] %(levelname)s : %(message)s', level=logging.INFO)
    extract_mid_stage_label_dataset('dataset/annoted_data.json')
