# -*- coding: utf-8 -*-
import json
from xml.dom import minidom


def read_json_formatted(dataset_filename):
    with open(dataset_filename) as data_file:
        data = json.load(data_file)
    return data


def read_absa_2015_restaurant_xml(filename):
    DOMTree = minidom.parse(filename)
    reviews = DOMTree.getElementsByTagName('Reviews')
    list_of_meta_dict = []
    for review in reviews:
        sentences = review.getElementsByTagName('sentence')
        for sentence in sentences:
            text_node = sentence.getElementsByTagName('text')[0]
            text = text_node.childNodes[0].nodeValue
            for opinion_dom in sentence.getElementsByTagName('Opinions'):
                opinion_list = opinion_dom.getElementsByTagName('Opinion')
                target_list = []
                polarity_list = []
                for opinion in opinion_list:
                    target_list.append(opinion.getAttribute('target'))
                    polarity_list.append(opinion.getAttribute('polarity'))

                d = {
                    'sentence': text,
                    'target':   target_list,
                    'polarity': polarity_list
                    }
                list_of_meta_dict.append(d)
    filename_json = filename.replace('.xml', '.json')
    if filename_json == filename:
        filename_json = filename + '.json'

    with open(filename_json, 'w') as outfile:
        json.dump(list_of_meta_dict, outfile)

    return filename_json
