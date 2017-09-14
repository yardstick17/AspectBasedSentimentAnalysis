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
MATCH_THRESHOLD = 0.2
ONLY_ASPECT_PREDICTION = False
POLARITY_ONLY_TASK = True


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
    # mid_training_label = [0] * len(list_of_extracted_meta)
    for i in range(1, 3):
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
                mid_training_label = [0] * len(list_of_extracted_meta)
                max_match_percent = match_percent
                max_match_extracted = extracted_tags
                for i_combination in combination:
                    if len(list_of_extracted_meta[i_combination]) > 0:
                        mid_training_label[i_combination] = 1

    return mid_training_label, max_match_extracted


def extract_mid_stage_label_dataframe(dataset_filename):
    """

    :param data-set_filename:
    :return:
    """

    if type(dataset_filename) == str:
        logging.info('Dataset: {}'.format(dataset_filename))
        annotated_dataset = get_dataset(dataset_filename)
    else:
        annotated_dataset = []
        logging.info('Datasets : {}'.format(', '.join(f for f in dataset_filename)))
        for dset in dataset_filename:
            annotated_dataset.extend(get_dataset(dset))

    initialize_globals()

    sorted_grammar_list = get_grammar()

    mid_training_data = []
    Y_PRED = []
    Y_TRUE = []

    for row in tqdm(annotated_dataset):
        sentence = row['sentence']
        logging.debug('sentence: ' + sentence)
        meta = {key: value for key, value in row['meta'].items() if key != 'null'}
        expected_meta_form = set(sorted(meta.items()))
        ste = SourceTargetExtractor(sentence)
        list_of_extracted_meta = list()
        for index, (_, compiled_grammar) in enumerate(sorted_grammar_list):
            score_dict = ste.get_topic_sentiment_score_dict(compiled_grammar)
            extracted_meta = get_polarity_form_result(score_dict)
            extracted_ote = set(extracted_meta.items())
            list_of_extracted_meta.append(extracted_ote)
        mid_training_label, max_match_extracted = get_max_combination(list_of_extracted_meta, expected_meta_form)
        y_pred_index, y_true_index = get_y_pred_and_y_true_label(expected_meta_form, max_match_extracted)

        Y_TRUE.extend(y_true_index)
        Y_PRED.extend(y_pred_index)

        mid_training_data.append([sentence, meta, max_match_extracted, mid_training_label])
    print('For Data-set: ', dataset_filename, '\n', classification_report(Y_TRUE, Y_PRED))
    df = pd.DataFrame(mid_training_data, columns=['sentence', 'meta', 'max_match_extracted', 'y_true'])

    # df.to_csv(MID_TRAINING_DATASET.format(dataset_filename.split('.')[0]))

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


def get_y_pred_and_y_true_label(expected_meta_form, extracted_meta_form, verbose=False):
    """

    :param expected_meta_form:
    :param extracted_meta_form:
    :return:
    """
    if verbose:
        # verbose = True
        polarity_default_dict = dict(extracted_meta_form)
        default_dict = dict(expected_meta_form)
        tmp_result_dict = {key: 'positive' for key, value in default_dict.items()}
        for key, value in tmp_result_dict.items():
            if key in polarity_default_dict:
                tmp_result_dict[key] = polarity_default_dict[key]

        extracted_meta_form = set(tmp_result_dict.items())

        print('expected_meta_form:', expected_meta_form)
        print('extracted_meta_form:', extracted_meta_form)
        print('intersection:', extracted_meta_form & expected_meta_form)
        print('false_negatives:', expected_meta_form - extracted_meta_form)
        print('false_positives:', extracted_meta_form - expected_meta_form)

    y_true_index = []
    y_pred_index = []

    intersecion_keys = set()
    # ext_dict = dict(extracted_meta_form)
    # exp_dict = dict(expected_meta_form)

    if not POLARITY_ONLY_TASK and ONLY_ASPECT_PREDICTION:
        extracted_meta_form = {i[0] for i in extracted_meta_form}
        expected_meta_form = {i[0] for i in expected_meta_form}
    else:
        ext_keys = {i[0] for i in extracted_meta_form}
        exp_keys = {i[0] for i in expected_meta_form}
        intersecion_keys = ext_keys & exp_keys

    intersection = len(extracted_meta_form & expected_meta_form)

    y_pred_index.extend([1] * intersection)
    y_true_index.extend([1] * intersection)


    common_removed_expected_meta_form = set()
    for item in expected_meta_form:
        if not item[0] in intersecion_keys:
            common_removed_expected_meta_form.add(item)

    common_removed_extracted_meta_form = set()
    for item in extracted_meta_form:
        if not item[0] in intersecion_keys:
            common_removed_extracted_meta_form.add(item)


    false_positives = len(extracted_meta_form - expected_meta_form)
    y_pred_index.extend([1] * false_positives)
    y_true_index.extend([0] * false_positives)

    expected_meta_form = common_removed_expected_meta_form

    false_negatives = len(expected_meta_form - extracted_meta_form)
    y_pred_index.extend([0] * false_negatives)
    y_true_index.extend([1] * false_negatives)


    # extracted_meta_form = common_removed_extracted_meta_form




    return y_pred_index, y_true_index
