import os

import pandas as pd

from dataset.read_dataset import read_json_formatted
from grammar.chunker import Chunker
from grammar.pattern_grammar import PatternGrammar


def initialize_globals():
    PatternGrammar().compile_all_source_target_grammar()
    PatternGrammar().compile_all_syntactic_grammar()

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
    initialize_globals()
    annoted_data_dataset = get_dataset()


    grammar_indices = PatternGrammar().syntactic_grammars.keys()

    for row in annoted_data_dataset:
        sentence = row['sentence']
        meta = row['meta']
        for index in grammar_indices:
            grammar = PatternGrammar().compile_syntactic_grammar(index)
            chunk_dict = Chunker(grammar).chunk_sentence(sentence)
            print(chunk_dict)




    pass
if __name__ == '__main__':
    process()
