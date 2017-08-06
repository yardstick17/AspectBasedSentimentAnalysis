# -*- coding: utf-8 -*-
import logging

import click
import numpy as np
from sklearn.metrics import classification_report

from feature_extraction.feature_vector_builder import get_syntactic_grammar_feature
from training.mid_stage_prepare_dataset import extract_mid_stage_label_dataframe
from training.mid_stage_prepare_dataset import get_syntactic_rules_in_list

LABEL = 'training_label'
SYNTACTIC_FEATURE = 'syntactic_feature'
from sklearn import svm

syntactic_rules_in_list = None

training_data = 'dataset/annoted_data.json'
testing_data = 'dataset/ABSA15_Restaurants_Test.json'


def get_syntactic_feature(row):
    """

    :param row:
    :return:
    """
    return get_syntactic_grammar_feature(row.sentence)


def transform_to_label(row):
    """

    :param row:
    :return:
    """
    grammar_label = get_syntactic_rules_in_list()
    return [i for i, j in enumerate(grammar_label) if j == row.label][0]


def get_features_and_label(dataset):
    """

    :param dataset:
    :return:
    """
    df = extract_mid_stage_label_dataframe(dataset)
    X = df.apply(get_syntactic_feature, axis=1)
    Y = df.apply(transform_to_label, axis=1)
    X = np.array(X.tolist())
    Y = np.array(Y.tolist())
    logging.info('Shape of array for dataset:  X:{} , Y:{} '.format(X.shape, Y.shape))
    return X, Y


@click.command()
@click.option('--log', '-l', help='set log level for the processing', default='INFO')
def main(log):
    """
    This method extracts the feature-vector and corresponding label for top-level classifier. The classifier is targeted
    to learn which rules to apply on a sentence so that correct opinion target extraction is done.
    :param log:
    """
    logging.basicConfig(format='[%(name)s] [%(asctime)s] %(levelname)s : %(message)s', level=logging._nameToLevel[log])
    X, Y = get_features_and_label(training_data)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X, Y)
    y_pred = classifier.predict(X)
    print('Classification report on training data\n', classification_report(Y, y_pred))

    X, Y = get_features_and_label(testing_data)
    y_pred = classifier.predict(X)

    print('Classification report on testing_data\n', classification_report(Y, y_pred))


if __name__ == '__main__':
    main()
