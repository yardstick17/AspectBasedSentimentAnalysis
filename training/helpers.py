# -*- coding: utf-8 -*-
import os

from tqdm import tqdm


def makedirs_with_mode(path, mode=0o775):
    try:
        old_mask = os.umask(0)
        os.makedirs(path, mode)
    finally:
        os.umask(old_mask)


def format_dataset(annotated_dataset):
    dataset = []
    for row in tqdm(
            annotated_dataset,
            total=len(annotated_dataset),
            unit='Reading Section'):
        sources = [s.lower() for s in row['target']]
        targets = [s.lower() for s in row['polarity']]
        sentence_meta = {}
        sentence = row['sentence']
        for source, target in zip(sources, targets):
            sentence_meta[source] = target
        dataset.append({'sentence': sentence, 'meta': sentence_meta})
    return dataset
