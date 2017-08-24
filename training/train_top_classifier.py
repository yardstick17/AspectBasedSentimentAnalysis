# -*- coding: utf-8 -*-
import logging

import click
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVC
from tqdm import tqdm

from feature_extraction.feature_vector_builder import get_syntactic_grammar_feature_vector
from grammar.source_target_extractor import SourceTargetExtractor
from training.mid_stage_prepare_dataset import extract_mid_stage_label_dataframe
from training.mid_stage_prepare_dataset import get_dataset
from training.mid_stage_prepare_dataset import get_grammar
from training.mid_stage_prepare_dataset import get_polarity_form_result
from training.mid_stage_prepare_dataset import get_y_pred_and_y_true_label
from training.mid_stage_prepare_dataset import initialize_globals

# FOREST = RandomForestClassifier(n_estimators=100, random_state=1)
FOREST = SVC(kernel='linear')
MULTI_TARGET_FOREST = MultiOutputRegressor(FOREST, n_jobs=-1)
LABEL = 'training_label'
SYNTACTIC_FEATURE = 'syntactic_feature'
CLASSIFIER_PKL = 'classifier.pkl'
syntactic_rules_in_list = None

training_data = ['dataset/annoted_data.json',
                 'dataset/customer_review_data/Apex AD2600 Progressive-scan DVD player.txt.json',
                 'dataset/customer_review_data/Nokia 6610.txt.json',
                 'dataset/customer_review_data/Creative Labs Nomad Jukebox Zen Xtra 40GB.txt.json',
                 ]
training_data = training_data[0]

# testing_data = 'dataset/customer_review_data/Canon G3.txt.json'
testing_data = 'dataset/ABSA15_Restaurants_Test.json'


# testing_data = 'dataset/ABSA15_Restaurants_Test.json'


# testing_data = 'dataset/customer_review_data/Apex AD2600 Progressive-scan DVD player.txt.json'
# testing_data = training_data


def get_syntactic_feature(row):
    """

    :param row:
    :return:
    """
    return get_syntactic_grammar_feature_vector(row.sentence)


def transform_to_label(row):
    """
    If the classifier expects label array of specific type, reformat here.

    :param row:
    :return:
    """
    return row.y_true


def get_features_and_label(dataset):
    """

    :param dataset:
    :return:
    """

    dataframe = pd.DataFrame()
    if type(dataset) == str:
        dataframe = extract_mid_stage_label_dataframe(dataset)
    else:
        dataframes = []
        for dset in dataset:
            print(dataset)
            dataframes.append(extract_mid_stage_label_dataframe(dset))
            dataframe = pd.concat(dataframes)

    X = dataframe.apply(get_syntactic_feature, axis=1)
    Y = dataframe.apply(transform_to_label, axis=1)
    X = np.array(X.tolist())
    Y = np.array(Y.tolist())
    logging.info('Shape of array for dataset:  X:{} , Y:{} '.format(X.shape, Y.shape))
    return X, Y, dataframe


@click.command()
@click.option('--log', '-l', help='set log level for the processing', default='INFO')
def main(log):
    """
    This method extracts the feature-vector and corresponding label for top-level classifier. The classifier is targeted
    to learn which rules to apply on a sentence so that correct opinion target extraction is done.
    :param log:
    """
    logging.basicConfig(format='[%(name)s] [%(asctime)s] %(levelname)s : %(message)s', level=logging._nameToLevel[log])
    X, Y, _ = get_features_and_label(training_data)
    columns_to_delete, Y = get_valid_columns(Y)
    classifier = MULTI_TARGET_FOREST
    classifier.fit(X, Y)
    y_pred = classifier.predict(X)

    print('Classification report on training data\n', classification_report(Y, y_pred))

    X, Y, test_dataframe = get_features_and_label(testing_data)
    Y = np.delete(Y, columns_to_delete, axis=1)
    y_pred = classifier.predict(X)
    print('Classification report on testing_data\n', classification_report(Y, y_pred))
    pd.to_pickle(classifier, CLASSIFIER_PKL)
    check_validity(testing_data, y_pred, columns_to_delete)


def get_valid_columns(X):
    X = np.array(X)
    col = X.shape[1]
    columns_to_delete = []
    for i in range(col):
        if not any(X[:, i]) or not any(map(lambda x: x != 1, X[:, i])):
            columns_to_delete.append(i)

    print('columns_to_delete', columns_to_delete)
    return columns_to_delete, np.delete(X, columns_to_delete, axis=1)


def check_validity(dataset_filename, y_pred, columns_to_delete):
    """

    :param dataset_filename:
    :return:
    """
    logging.info('Dataset: {}'.format(dataset_filename))
    initialize_globals()
    annoted_data_dataset = get_dataset(dataset_filename)
    sorted_grammar_list = get_grammar()
    sorted_grammar_list = [grammar for index, grammar in enumerate(sorted_grammar_list) if
                           index not in columns_to_delete]

    Y_PRED = []
    Y_TRUE = []
    for row, pred in tqdm(zip(annoted_data_dataset, y_pred)):
        sentence = row['sentence']
        # meta = row['meta']
        meta = {key: value for key, value in row['meta'].items() if key != 'null'}
        expected_meta_form = set(meta.keys())
        ste = SourceTargetExtractor(sentence)
        overall_extracted_meta = set()
        for index, rule_flag in enumerate(pred):
            if rule_flag == 1:
                compiled_grammar = sorted_grammar_list[index][1]
                score_dict = ste.get_topic_sentiment_score_dict(compiled_grammar)
                extracted_meta = get_polarity_form_result(score_dict)
                overall_extracted_meta.update(extracted_meta.keys())

        y_pred_index, y_true_index = get_y_pred_and_y_true_label(expected_meta_form,
                                                                 overall_extracted_meta)
        Y_TRUE.extend(y_true_index)
        Y_PRED.extend(y_pred_index)

    print('::::::::::::::::::   TESTING   ::::::::::::::::::\n', dataset_filename, '\n',
          classification_report(Y_TRUE, Y_PRED))


if __name__ == '__main__':
    main()
