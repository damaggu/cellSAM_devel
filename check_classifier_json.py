import json

import numpy as np
import pandas as pd


def reverse_dict(dict):
    reversed_dict = {}
    for key, value in dict.items():
        if isinstance(value, list):
            for item in value:
                reversed_dict[item] = key
        else:
            reversed_dict[value] = key
    return reversed_dict


def compute_acc(dict, annotations):
    correct_cnt = 0
    for img in list(annotations[0]):
        
        pred = dict[img]
        if pred == 'no_wsi':
            pred = 1
        elif pred == 'flagged':
            pred = 1
        elif pred == 'regular':
            pred = 0

        gt = annotations[annotations[0] == img][1].values[0]

        if gt == pred:
            correct_cnt += 1

    acc = correct_cnt / len(list(annotations[0]))

    return acc

if __name__ == "__main__":

    # ANNOTATION TYPE
    ANN_TYPE = 'tune'  # 'tune' or 'hidden'

    # get annotation values
    ## load labels from csv files
    tune_csv_file = 'bloodcellSheet_tuning.csv'
    hidden_csv_file = 'bloodcellSheet_hidden.csv'

    tune_annotations = pd.read_csv(tune_csv_file, header=None)
    tune_annotations[1] = pd.to_numeric(tune_annotations[1], errors='coerce').fillna(0).astype(int)
    tune_annotations[0] = tune_annotations[0].str.split('.').str[0]
    tune_annotations = tune_annotations.sort_values(by=0, axis=0)

    hidden_annotations = pd.read_csv(hidden_csv_file, header=None)
    hidden_annotations[1] = pd.to_numeric(hidden_annotations[1], errors='coerce').fillna(0).astype(int)
    hidden_annotations[0] = hidden_annotations[0].str.split('.').str[0]
    hidden_annotations = hidden_annotations.sort_values(by=0, axis=0)

    # load the json file
    joson_path_root = 'tmp_outs/'
    json_file_name = 'wsi_imgs_dict.json'

    # json_folder = 'results_hidden_test2'
    # json_folder = 'results_tune_test2_nopreproc'
    json_folder = 'results_tune_test2'
    json_path = joson_path_root + json_folder + '/' + json_file_name
    with open(json_path) as f:
        data = json.load(f)

    # get the reverse dict
    reversed_dict = reverse_dict(data)
    reverse_dict = {key.split('.')[0]: value for key, value in reversed_dict.items()}

    # compute the accuracy
    if ANN_TYPE == 'tune':
        acc = compute_acc(reverse_dict, tune_annotations)
    elif ANN_TYPE == 'hidden':
        acc = compute_acc(reverse_dict, hidden_annotations)
    else:
        raise ValueError('ANN_TYPE should be either tune or hidden')
    
    print(f'{ANN_TYPE} accuracy: {acc}')

    print('htest')