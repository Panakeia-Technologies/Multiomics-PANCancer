"""
Copyright (c) 2022- Panakeia Technologies Limited
Software licensed under GNU General Public License (GPL) version 3

Module defining standard evaluation metrics and evaluation functions.
"""

import warnings
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import f1_score as f1_score_fn

from transforms import average_fn, majority_fn


class BinaryClassificationEvaluator:   
    """
    Class for computing evaluation metrics for a binary classification problem.
    """

    @staticmethod
    def evaluate(results):
        """
        Compute a bunch of standard evaluation metrics based on model predictions.

        :param results: (dict) the following information are taken into consideration
            - 'y_true': (list-like, int) targets, i.e. true values
            - 'y_pred': (list-like, int) predictions
            - 'y_score': (list-like, float) scores of positive class
            - 'loss': (float) loss; optional
        :return: (dict) the computed metrics
        """

        if 'y_true' not in results or 'y_pred' not in results or 'y_score' not in results:
            raise ValueError("Cannot call BinaryClassificationEvaluator since at least one of 'y_true', 'y_pred', "
                             "'y_score' has not been provided in the results dict.")

        metrics = get_eval_metrics(results['y_true'], results['y_pred'], results['y_score'])

        if 'loss' in results:
            metrics['loss'] = results['loss']

        print(metrics)

        return metrics


def get_eval_metrics(y_true, y_pred, y_score, average='binary'):
    """
    Standard evaluation metrics for a list of true and predicted values.

    :param y_true: (list) list of true values
    :param y_pred (list) list of predicted values
    :param y_score (list) list of model prediction scores
    :param average (str) determines the type of averaging performed on the data
    :return (dict) of evaluation metrics
    """

    assert all(val in [0, 1] for val in y_true)
    assert all(val in [0, 1] for val in y_pred)
    assert all(0 <= val <= 1 for val in y_score)

    acc = accuracy_score(y_true, y_pred)

    with warnings.catch_warnings(record=True) as warn:
        f1_score = f1_score_fn(y_true, y_pred, average=average)
        f1_score = f1_score if len(warn) == 0 else np.nan

    auc_val = roc_auc(y_true, y_score)

    metrics = {
        'accuracy': acc,
        'f1': f1_score,
        'auc': auc_val,
    }

    if len(y_true) > 0:
        conf = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = conf.ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        sens = 1 - fnr
        spec = 1 - fpr
    else:
        fpr, fnr, ppv, npv, sens, spec = [np.nan] * 10

    metrics.update(
        {
            'fpr': fpr,
            'fnr': fnr,
            'ppv': ppv,
            'npv': npv,
            'sens': sens,
            'spec': spec,
        }
    )

    return metrics


def roc_auc(y_true, y_score, pos_label=1):
    """
    Compute area under curve (AUC). Only supports binary classification.

    :param y_true: (list) list of true values
    :param y_score (list) list of probabilities
    :param pos_label (int) the label of the positive class
    :return AUC value
    """
    assert all(val in [0, 1] for val in y_true)
    assert all(0 <= val <= 1 for val in y_score)
    assert len(y_true) == len(y_score)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
        return auc(fpr, tpr)
    except Exception:
        return np.nan


def evalaute_results(results, image_to_patient_lookup):
    """
    Function for evaluating results.

    :param results: (ImageLevelEstimatorResults) from validation
    :param image_to_patient_lookup: (dict) mapping between images and patients
    """
    
    evaluator = BinaryClassificationEvaluator()
    results = results.get_results_dict()
    metrics = {}
    print('Image level metrics')
    metrics['image_level'] = evaluator.evaluate(results)
    print('Patient level metrics')
    patient_results = compute_patient_results(results, image_to_patient_lookup)
    metrics['patient_level'] = evaluator.evaluate(patient_results)
    return metrics
    
    
def compute_patient_results(results, image_to_patient_lookup):
    """
    Computes patient level metrics and returns them as a dictionary.

    :param results: (dict) image level results dictionary containing keys including 'image_ids', 'y_score',
        'y_pred' and 'y_true'
    :param image_to_patient_lookup: (dict) Lookup for image-to-patient mapping
    :return: (dict) patient level metrics
    """
   
    # Group results by patients and evaluate
    grouped_results = group_results(results, image_to_patient_lookup)
    patient_level_results = pool_results(grouped_results)    
    return patient_level_results


def group_results(image_level_results, image_to_patient_lookup):
    """
    Groups the results at image level to patient based on image_to_patient_lookup.

    :param image_level_results: (dict) Image-level results
    :param image_to_patient_lookup: (dict) Lookup for mapping images to patients
    :return: (dict<str, list<tuple>) of the form {attribute_val: (score, pred, true), ...], ...}
    """
        
    image_ids = image_level_results['image_ids']
    y_score = image_level_results['y_score']
    y_pred = image_level_results['y_pred']
    y_true = image_level_results['y_true']
    
    grouped_results = {}

    for image_uuid, score, pred, true in zip(image_ids, y_score, y_pred, y_true):
        if image_uuid in image_to_patient_lookup:
            patient_uuid = image_to_patient_lookup[image_uuid]
            result = (score, pred, true)
            grouped_results.setdefault(patient_uuid, []).append(result)

    return grouped_results


def pool_results(grouped_results):
    """
    Aggregates the results from one level to the next.

    :param grouped_results: (dict<str, list<tuple>) of the form {uuid: [(score, pred, true), ...], ...}
    :return: (dict<str, tuple) of the form {uuid: (score, pred, true), ...}
    """

    pooled_results = {}
    patient_level_results = {}
    for uuid, results in grouped_results.items():
        y_scores, y_preds, y_trues = list(zip(*results))
        score = average_fn(y_scores)
        pred = majority_fn(y_preds)
        true = y_trues[0]
        pooled_results[uuid] = (score, pred, true)
    
    y_score, y_pred, y_true = list(zip(*pooled_results.values()))
    patient_level_results['y_score'] = list(y_score)
    patient_level_results['y_pred'] = list(y_pred)
    patient_level_results['y_true'] = list(y_true)

    return patient_level_results
