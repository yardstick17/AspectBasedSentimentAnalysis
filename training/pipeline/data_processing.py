# -*- coding: utf-8 -*-
import logging

import luigi
import numpy as np
import pandas as pd
from tqdm import tqdm

from grammar.source_target_extractor import SourceTargetExtractor
from training.mid_stage_prepare_dataset import get_grammar
from training.mid_stage_prepare_dataset import get_max_combination
from training.mid_stage_prepare_dataset import get_polarity_form_result
from training.mid_stage_prepare_dataset import get_y_pred_and_y_true_label
from training.mid_stage_prepare_dataset import initialize_globals
from training.pipeline.acquire_dataset import AcquireDataset
from training.pipeline.acquire_dataset import BaseTask
from training.train_top_classifier import get_syntactic_feature
from training.train_top_classifier import transform_to_label

logger = logging.getLogger(__name__)


def process_data_for_training(annotated_dataset):
    sorted_grammar_list = get_grammar()
    mid_training_data = []
    Y_PRED = []
    Y_TRUE = []
    for row in tqdm(annotated_dataset):
        sentence = row['sentence']
        logging.debug('sentence: ' + sentence)
        meta = {
            key: value
            for key, value in row['meta'].items() if key != 'null'
        }
        expected_meta_form = set(sorted(meta.items()))
        ste = SourceTargetExtractor(sentence)
        list_of_extracted_meta = list()
        for index, (_, compiled_grammar) in enumerate(sorted_grammar_list):
            score_dict = ste.get_topic_sentiment_score_dict(compiled_grammar)
            extracted_meta = get_polarity_form_result(score_dict)
            extracted_ote = set(extracted_meta.items())
            list_of_extracted_meta.append(extracted_ote)
        mid_training_label, max_match_extracted = get_max_combination(
            list_of_extracted_meta, expected_meta_form)
        y_pred_index, y_true_index = get_y_pred_and_y_true_label(
            expected_meta_form, max_match_extracted)
        Y_TRUE.extend(y_true_index)
        Y_PRED.extend(y_pred_index)
        mid_training_data.append(
            [sentence, meta, max_match_extracted, mid_training_label])
    dataframe = pd.DataFrame(
        mid_training_data,
        columns=['sentence', 'meta', 'max_match_extracted', 'y_true'])
    X = dataframe.apply(get_syntactic_feature, axis=1)
    Y = dataframe.apply(transform_to_label, axis=1)
    X = np.array(X.tolist())
    Y = np.array(Y.tolist())
    logging.info(
        'Shape of array for dataset:  X:{} , Y:{} '.format(X.shape, Y.shape))
    td = TrainingData(X, Y)
    return td


class TrainingData(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y


class DataProcessing(BaseTask):
    dataset_filename = luigi.Parameter()

    def run(self):
        initialize_globals()
        annotated_dataset = self.get_annotated_data()

        td = process_data_for_training(annotated_dataset)
        pd.to_pickle(td, self.output().path)

    def get_annotated_data(self):
        filepath = self.requires().output().path
        return pd.read_pickle(filepath)

    def requires(self):
        return AcquireDataset(self.dataset_filename)


if __name__ == '__main__':
    luigi.run()
