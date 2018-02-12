# -*- coding: utf-8 -*-
import logging

import luigi
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVC

from training.pipeline.acquire_dataset import BaseTask
from training.pipeline.data_processing import DataProcessing

logger = logging.getLogger(__name__)


class TrainRuleFitnessClassifier(BaseTask):
    dataset_filename = luigi.Parameter()

    def run(self):
        training_data = self.get_features_and_label()
        classifier = self.get_multi_label_classifier()
        classifier.fit(training_data.X, training_data.Y)
        pd.to_pickle(classifier, self.output().path)

    def get_multi_label_classifier(self):
        forest = SVC(kernel='linear')
        return MultiOutputRegressor(forest, n_jobs=-1)

    def get_features_and_label(self):
        td = pd.read_pickle(self.requires().output().path)
        return td

    def requires(self):
        return DataProcessing(self.dataset_filename)


if __name__ == '__main__':
    luigi.run()
