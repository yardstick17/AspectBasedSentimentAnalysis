# -*- coding: utf-8 -*-
import json
import re
from xml.dom import minidom


def read_json_formatted(dataset_filename):
    with open(dataset_filename) as data_file:
        data = json.load(data_file)
    return data


def read_absa_2014_restaurant_xml(filename):
    """

    :param filename:
    :return:
    """
    import ipdb
    ipdb.set_trace()
    DOMTree = minidom.parse(filename)
    reviews = DOMTree.getElementsByTagName('sentences')
    list_of_meta_dict = []
    for review in reviews:
        sentences = review.getElementsByTagName('sentence')
        for sentence in sentences:
            text_node = sentence.getElementsByTagName('text')[0]
            text = text_node.childNodes[0].nodeValue
            for opinion_dom in sentence.getElementsByTagName('aspectTerms'):
                opinion_list = opinion_dom.getElementsByTagName('aspectTerm')
                target_list = []
                polarity_list = []
                for opinion in opinion_list:
                    target_list.append(opinion.getAttribute('term').strip())
                    polarity_list.append(opinion.getAttribute('polarity').strip())

                d = {
                    'sentence': text,
                    'target': target_list,
                    'polarity': polarity_list
                }
                list_of_meta_dict.append(d)
    filename_json = filename.replace('.xml', '.json')
    if filename_json == filename:
        filename_json = filename + '.json'

    with open(filename_json, 'w') as outfile:
        json.dump(list_of_meta_dict, outfile)

    return filename_json


def read_absa_2015_restaurant_xml(filename):
    """

    :param filename:
    :return:
    """
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
                    target_list.append(opinion.getAttribute('target').strip())
                    polarity_list.append(opinion.getAttribute('polarity').strip())

                d = {
                    'sentence': text,
                    'target': target_list,
                    'polarity': polarity_list
                }
                list_of_meta_dict.append(d)
    filename_json = filename.replace('.xml', '.json')
    if filename_json == filename:
        filename_json = filename + '.json'

    with open(filename_json, 'w') as outfile:
        json.dump(list_of_meta_dict, outfile)

    return filename_json


def read_customer_review_data(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    list_of_meta_dict = []
    for line in content:
        if '[t]' not in line and '[p]' not in line and '[cs]' not in line \
                and '[cc]' not in line and '[s]' not in line and '[u]' not in line:
            splitted_line = line.split('##')
            print(splitted_line, line)
            sentence = splitted_line[1]
            aspects_with_polarity = splitted_line[0].split(',')
            aspect_list = []
            polarity_list = []

            for aspect in aspects_with_polarity:
                if aspect:
                    # aspect_list.append(aspect)
                    stripped_aspect = re.sub('[^A-Za-z\s]+', '', aspect).strip()
                    if '+' in aspect:
                        aspect_list.append(stripped_aspect)
                        polarity_list.append('positive')
                    elif '-' in aspect:
                        aspect_list.append(stripped_aspect)
                        polarity_list.append('negative')
            d = {
                'sentence': sentence,
                'target': aspect_list,
                'polarity': polarity_list
            }
            list_of_meta_dict.append(d)

    filename_json = filename.replace('.xml', '.json')
    if filename_json == filename:
        filename_json = filename + '.json'

    with open(filename_json, 'w') as outfile:
        json.dump(list_of_meta_dict, outfile)

    return filename_json
