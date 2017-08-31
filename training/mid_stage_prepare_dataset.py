# -*- coding: utf-8 -*-
import itertools
import logging

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset.read_dataset import read_json_formatted
from grammar.pattern_grammar import PatternGrammar
from grammar.source_target_extractor import SourceTargetExtractor

LABEL_LIST_PKL = 'label_list.pkl'
MID_TRAINING_DATASET = '{}_mid_training_data.csv'
grammar_label = None
SYNTACTIC_COMPILED_GRAMMAR_PKL_FILE = 'label_referenced_syntactic_compiled_grammar.pkl'
MATCH_THRESHOLD = 0.1


def initialize_globals():
    """
    This method compiles the grammar and cache it in the global variable.

    """
    PatternGrammar().compile_all_source_target_grammar()
    PatternGrammar().compile_all_syntactic_grammar()


def dataset_expanded(dataset_filename):
    """

    :param data-set_filename: Filename of the dataset to be read
    :return: return the pandas DataFrame with columns: [sentence,target,opinion]
    """
    annotated_data = read_json_formatted(dataset_filename)
    dataset = []
    for row in annotated_data:
        sources = [s.lower() for s in row['target']]
        targets = [s.lower() for s in row['polarity']]
        sentence_meta = {}
        sentence = row['sentence']
        for source, target in zip(sources, targets):
            sentence_meta[source] = target
            dataset.append([sentence, source, target])

    return pd.DataFrame(dataset, columns='sentence,target,opinion'.split(','))


def get_dataset(dataset_filename=None):
    """

    :param data-set_filename:
    :return:
    """
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
    """

    :param score:
    :return:
    """
    return 'negative' if score['PosScore'] < score['NegScore'] else 'positive'


def get_syntactic_rules_in_list():
    """

    :return:
    """
    global grammar_label
    if grammar_label is None:
        grammar_label = pd.read_pickle(LABEL_LIST_PKL)
    return grammar_label


def get_max_combination(list_of_extracted_meta, expected_meta_form):
    """

    :param list_of_extracted_meta:
    :param expected_meta_form:
    :return:
    """
    total_rules = list(range(len(list_of_extracted_meta)))
    mid_training_label = [0] * len(list_of_extracted_meta)
    max_match_extracted = set()
    max_match_percent = 0
    for i in range(1, 4):
        all_combinations = list(itertools.combinations(total_rules, i))
        for combination in all_combinations:
            combination = list(combination)
            extracted_tags = set()
            for index in combination:
                extracted_tags.update(list_of_extracted_meta[index])
            y_true_index, y_pred_index = get_y_pred_and_y_true_label(expected_meta_form,
                                                                     extracted_tags)

            match_percent = f1_score(y_true_index, y_pred_index)
            if match_percent >= MATCH_THRESHOLD and match_percent > max_match_percent:
                max_match_percent = match_percent
                max_match_extracted = extracted_tags
                mid_training_label = [0] * len(list_of_extracted_meta)
                for i_combination in combination:
                    if len(list_of_extracted_meta[i_combination]) > 0:
                        mid_training_label[i_combination] = 1

    return mid_training_label, max_match_extracted


def extract_mid_stage_label_dataframe(dataset_filename):
    """

    :param data-set_filename:
    :return:
    """
    logging.info('Dataset: {}'.format(dataset_filename))
    initialize_globals()
    annoted_data_dataset = get_dataset(dataset_filename)
    sorted_grammar_list = get_grammar()

    baseline_ote = set()
    for row in tqdm(annoted_data_dataset):
        meta = {key: value for key, value in row['meta'].items() if key != 'null'}
        baseline_ote.update(meta.keys())

    mid_training_data = []
    Y_PRED = []
    Y_TRUE = []
    for row in tqdm(annoted_data_dataset):
        sentence = row['sentence']
        base_otes = baseline_ote & set(sentence.split())

        meta = {key: value for key, value in row['meta'].items() if key != 'null'}
        expected_meta_form = set(sorted(meta.keys()))
        ste = SourceTargetExtractor(sentence)
        list_of_extracted_meta = list()

        for index, (_, compiled_grammar) in enumerate(sorted_grammar_list):
            score_dict = ste.get_topic_sentiment_score_dict(compiled_grammar)
            extracted_meta = get_polarity_form_result(score_dict)
            extracted_ote = set(extracted_meta.keys())
            base_otes = extracted_ote | base_otes

            list_of_extracted_meta.append(extracted_ote)

        mid_training_label, max_match_extracted = get_max_combination(list_of_extracted_meta, expected_meta_form)
        # if 'Test' in dataset_filename:
        #     max_match_extracted = max_match_extracted | base_otes
        y_pred_index, y_true_index = get_y_pred_and_y_true_label(expected_meta_form,
                                                                 max_match_extracted)
        Y_TRUE.extend(y_true_index)
        Y_PRED.extend(y_pred_index)
        mid_training_data.append([sentence, meta, max_match_extracted, mid_training_label])
    print('For Data-set: ', dataset_filename, '\n', classification_report(Y_TRUE, Y_PRED))
    df = pd.DataFrame(mid_training_data, columns=['sentence', 'meta', 'max_match_extracted', 'y_true'])
    df.to_csv(MID_TRAINING_DATASET.format(dataset_filename.split('.')[0]))
    return df


def get_polarity_form_result(score_dict):
    """

    :param score_dict:
    :return:
    """
    extracted_meta = {}
    for source, score in score_dict.items():
        source = source.lower().strip()
        if source != '':
            if score['PosScore'] < score['NegScore']:
                extracted_meta[source] = 'negative'
            else:
                extracted_meta[source] = 'positive'
    return extracted_meta


def get_grammar():
    """

    :return:
    """
    syntactic_compiled_grammar = PatternGrammar().compile_all_syntactic_grammar()
    return sorted(syntactic_compiled_grammar.items(), key=lambda x: x, reverse=True)


def get_y_pred_and_y_true_label(expected_meta_form, extracted_meta_form):
    """

    :param expected_meta_form:
    :param extracted_meta_form:
    :return:
    """
    y_true_index = []
    y_pred_index = []

    intersection = len(extracted_meta_form & expected_meta_form)
    y_true_index.extend([1] * intersection)
    y_pred_index.extend([1] * intersection)

    false_negatives = len(extracted_meta_form - expected_meta_form)
    y_true_index.extend([1] * false_negatives)
    y_pred_index.extend([0] * false_negatives)

    false_positives = len(expected_meta_form - extracted_meta_form)
    y_true_index.extend([0] * false_positives)
    y_pred_index.extend([1] * false_positives)

    return y_pred_index, y_true_index
