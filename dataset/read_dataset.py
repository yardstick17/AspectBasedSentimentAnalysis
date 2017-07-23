import json


def read_json_formatted():
    # sentences = json.load('scripts/Grammer/annoted_sentences.json')
    with open('AspectBasedSentimentAnalysis/dataset/annoted_data.json') as data_file:
        data = json.load(data_file)
    return data
