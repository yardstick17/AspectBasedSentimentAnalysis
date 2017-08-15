# -*- coding: utf-8 -*-
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
MATCH_THRESHOLD = 40


def initialize_globals():
    """
    This method compiles the grammar and cache it in the global variable.

    """
    PatternGrammar().compile_all_source_target_grammar()
    PatternGrammar().compile_all_syntactic_grammar()


def dataset_expanded(dataset_filename):
    """

    :param dataset_filename: Filename of the dataset to be read
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

    :param dataset_filename:
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


def extract_mid_stage_label_dataframe(dataset_filename):
    """

    :param dataset_filename:
    :return:
    """
    logging.info('Dataset: {}'.format(dataset_filename))
    initialize_globals()
    annoted_data_dataset = get_dataset(dataset_filename)
    sorted_grammar_list = get_grammar()
    mid_training_data = []
    Y_PRED = []
    Y_TRUE = []
    for row in tqdm(annoted_data_dataset):
        sentence = row['sentence']
        meta = row['meta']
        expected_meta_form = set(sorted(meta.items()))
        ste = SourceTargetExtractor(sentence)
        y_pred_for_sentence = []
        best_result_from_rule = set()
        for index, (_, compiled_grammar) in enumerate(sorted_grammar_list):
            score_dict = ste.get_topic_sentiment_score_dict(compiled_grammar)
            extracted_meta = get_polarity_form_result(score_dict)
            extracted_meta_form = set(sorted(extracted_meta.items()))
            y_pred_index, y_true_index = get_y_pred_and_y_true_label(expected_meta_form, extracted_meta_form)
            match_percent = 1 if int(100 * f1_score(y_true_index, y_pred_index)) >= MATCH_THRESHOLD else 0
            if match_percent == 1:
                best_result_from_rule.update(extracted_meta_form)
            y_pred_for_sentence.append(match_percent)

        y_pred_index, y_true_index = get_y_pred_and_y_true_label(expected_meta_form, best_result_from_rule)
        Y_TRUE.extend(y_true_index)
        Y_PRED.extend(y_pred_index)

        mid_training_data.append([sentence, y_pred_for_sentence, best_result_from_rule])
    print('For Data-set: ', dataset_filename, '\n', classification_report(Y_TRUE, Y_PRED))
    df = pd.DataFrame(mid_training_data, columns=['sentence', 'y_true', 'best_result_from_rule'])
    df.to_csv(MID_TRAINING_DATASET.format(dataset_filename))
    return df


def get_polarity_form_result(score_dict):
    extracted_meta = {}
    for source, score in score_dict.items():
        source = source.lower()
        if score['PosScore'] <= score['NegScore']:
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
    intersection = extracted_meta_form & expected_meta_form
    y_true_index = []
    y_pred_index = []
    for _ in range(len(intersection)):
        y_true_index.append(1)
        y_pred_index.append(1)
    for _ in range(len(expected_meta_form - extracted_meta_form)):
        y_true_index.append(1)
        y_pred_index.append(0)
    for _ in range(len(extracted_meta_form - expected_meta_form)):
        y_true_index.append(0)
        y_pred_index.append(1)

    if len(y_true_index) == 0:
        y_true_index.append(0)
        y_pred_index.append(0)

    return y_pred_index, y_true_index


if __name__ == '__main__':
    logging.basicConfig(format='[%(name)s] [%(asctime)s] %(levelname)s : %(message)s', level=logging.INFO)
    extract_mid_stage_label_dataframe('dataset/annoted_data.json')
