"""
Copyright (c) 2022- Panakeia Technologies Limited
Software licensed under GNU General Public License (GPL) version 3

A module that defines transformations used for manipulating data during training.
"""

import numpy as np
import torch
from collections import Counter

DEFAULT_NORM_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORM_STD = [0.229, 0.224, 0.225]


class Normalizer:
    """
    A class to normalize (batched) data.

    :param mean: (array) mean array for the three channels, only used if backbone is None
    :param std: (array) std array for the three channels, only used if backbone is None
    :param is_cuda: (bool) whether to use CUDA for this transformation
    :param batched: (bool) whether input data come in batch format (batch size, C, H, W) or individually (C, H, W)
    """
    def __init__(self, mean=None, std=None, is_cuda=True, batched=True):
        self.is_cuda = is_cuda
        self.batched = batched
        self.mean = DEFAULT_NORM_MEAN if mean is None else mean
        self.std = DEFAULT_NORM_STD if std is None else std

    def __call__(self, input_tensor):
        """
        :param input_tensor: (Tensor) to be normalized; float array in [0,1] of shape (batch size, C, H, W) if batched
            is True, otherwise of shape (C, H, W)
        :return: (Tensor) Normalized tensor of same shape
        """

        # self.mean and self.std are tiled and converted to tensor, which is then reshaped to
        # (batch_size, num_channels, 1, 1), where num_channels is 3, so that normalisation can be performed
        # for each image in one go.
        if self.batched:
            batch_size = input_tensor.size(0)
            im_mean = torch.Tensor(np.tile(self.mean, (batch_size, 1))).view(batch_size, 3, 1, 1)
            im_std = torch.Tensor(np.tile(self.std, (batch_size, 1))).view(batch_size, 3, 1, 1)
        else:
            im_mean = torch.Tensor(self.mean).view(3, 1, 1)
            im_std = torch.Tensor(self.std).view(3, 1, 1)
        if self.is_cuda:
            return (input_tensor - im_mean.cuda()) / im_std.cuda()
        return (input_tensor - im_mean) / im_std


class ToTensor:
    """
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]"""

    def __init__(self, is_cuda=True):
        self.is_cuda = is_cuda

    def __call__(self, img):
        """
        :param img: (PIL.Image / numpy.ndarray) image of size (H, W, C) to be converted.
        :return: (Tensor) torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        """

        tensor = torch.FloatTensor(np.array(img) / 255.).permute(2, 0, 1)
        if self.is_cuda and torch.cuda.is_available():
            return tensor.cuda()
        return tensor


class Compose:
    """
    Composes multiple transformations.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        """

        :param img: image of size (H, W, C) to be transformed.
        :return: transformed image
        """

        for transform in self.transforms:
            img = transform(img)

        return img
    
    
def get_transformation(is_cuda, normalize=True):
    """
    Get transformation for training and validation (prediction) datasets.

    :param is_cuda: (bool) whether to use CUDA for selected transformation operations
    :param normalize: (bool) whether to apply a normalizer to the input patches
    :return tuple: composed transformations to be for training and prediction
    """

    transform = [ToTensor(is_cuda=is_cuda)]

    if normalize:
        transform += [Normalizer(is_cuda=is_cuda, batched=False)]

    return Compose(transform)


def majority_fn(targets):
    """
    Find the majority class in a list of targets

    :param targets: (list) of class predictions or targets
    :return: majority class
    """
    return Counter(sorted(targets, reverse=True)).most_common()[0][0]


def average_fn(targets, weights=None):
    """
    Computes average of a list. Can be weighted average if required.

    :param targets: (list) of values
    :param weights: (list?)
    :return: (float) weighted average of list of values
    """
    if weights is None:
        return sum(targets) / len(targets)
    else:
        assert len(targets) == len(weights), "values and weights lists must be of the same length"
        return sum([val * weight for val, weight in list(zip(targets, weights))]) / sum(weights)


def apply_decision(scores):
    """
    Apply decision threshold

    :param: (np.array) 1-d array of model output scores for each class with its length equal to `num_classes`
    :return: (int) predicted class
    """
    return np.argmax(scores)


