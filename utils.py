"""
Copyright (c) 2022- Panakeia Technologies Limited
Software licensed under GNU General Public License (GPL) version 3

Misc utility functions.
"""

import json
import os

import pandas as pd
import numpy as np
import torch
from torch.backends import cudnn

DEFAULT_FOLD_LIST = [0, 1, 2]


def set_reproducibility_seed(seed=42):
    """
    Function to make random functions reproducible.

    :param seed: (int) seed to be set to ensure deterministic experiment
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    print(f'Seed set to {seed}')


def parse_profile_file(profile_file_path, val_fold=0):
    """
    A function to read and parse a profile file to acquire image IDs, targets, cross-validation fold membership of
    each image, and patient IDs. Returns two lookup dictionaries, one for image-to-target mapping and one for
    image-to-patient mapping.
    :param profile_file_path: (str) Path to the profile file that has the following columns: ['image_id', 'target',
        'cv_fold', 'patient_id']
    :param val_fold: (int) Validation fold

    :return: (dict, dict):
        targets_lookup:  image-to-target mapping
        image_to_patient_lookup: image-to-patient mapping
    """

    # Read profile file
    profile_df = pd.read_csv(profile_file_path)

    # Get training and val folds
    training_folds = DEFAULT_FOLD_LIST
    val_folds = [val_fold]
    training_folds.remove(val_fold)

    targets_lookup = {
        'Training': {
            'image_ids': [],
            'targets': []
        },

        'Validation': {
            'image_ids': [],
            'targets': []
        }
    }

    image_to_patient_lookup = {}

    image_ids = profile_df['image_id'].tolist()
    targets = profile_df['target'].tolist()
    cv_folds = profile_df['cv_fold'].tolist()
    patient_ids = profile_df['patient_id'].tolist()

    for image_id, target, fold, patient_id in zip(image_ids, targets, cv_folds, patient_ids):
        if fold in val_folds:
            mode = 'Validation'
        elif fold in training_folds:
            mode = 'Training'
        else:
            raise ValueError(f'cv_fold can only be one of {DEFAULT_FOLD_LIST}')
        targets_lookup[mode]['image_ids'].append(image_id)
        targets_lookup[mode]['targets'].append(target)
        image_to_patient_lookup[image_id] = patient_id

    return targets_lookup, image_to_patient_lookup


def save_results_as_json(results_dict, save_dir, run_name, file_name):
    """
    Save results dict as json.
    """
    os.makedirs(os.path.join(save_dir, run_name), exist_ok=True)
    local_results_path = os.path.join(save_dir, run_name, f'{file_name}.json')

    with open(local_results_path, 'w') as fp:
        json.dump(results_dict, fp)
