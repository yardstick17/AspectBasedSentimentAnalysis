import json
import pandas as pd
import os

from dataset.read_dataset import read_json_formatted


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
    annoted_data_dataset = get_dataset()
    for row in annoted_data_dataset:
        sentence = row['sentence']
        meta = row['meta']

    pass
if __name__ == '__main__':
    pass
