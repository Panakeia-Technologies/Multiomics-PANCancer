"""
Copyright (c) 2022- Panakeia Technologies Limited
Software licensed under GNU General Public License (GPL) version 3

Module for representations and operations on single whole slide images.
"""

import torch
from transforms import apply_decision, average_fn


class Slide:
    """
    A container for a slide patched into tiles.

    :param tiles: (list<Tile>) Representation of all tiles.
    :param is_cuda: (bool) Whether slide's tensor attributes will be returned as cuda tensors
    """

    def __init__(self, tiles, is_cuda=False):
        self._tiles = tiles
        self._is_cuda = is_cuda
        self._score = None
        self._pred = None

    @property
    def tiles(self):
        """
        (list<Tile>)
        """
        return self._tiles

    @property
    def num_tiles(self):
        """
        (int) Number of tiles stored in the slide object.
        """
        return len(self._tiles)

    @property
    def tile_scores(self):
        """
        (torch.Tensor) List of scores assigned to the tiles of the slide. None if no scores have been assigned.
        """
        if all(tile.score is not None for tile in self._tiles):
            scores = torch.tensor([tile.score for tile in self._tiles])
            if self._is_cuda:
                scores = scores.cuda()
            return scores
        else:
            return None

    @property
    def is_cuda(self):
        """
        :return: (bool) Whether slide tensor properties are returned on CUDA
        """
        return self._is_cuda

    @is_cuda.setter
    def is_cuda(self, is_cuda):
        self._is_cuda = is_cuda

    @property
    def score(self):
        """
        (float?)
        """
        return self._score

    @property
    def pred(self):
        """
        (int?)
        """
        return self._pred

    def aggregate_tile_results(self):
        """
        Aggregates scores and predictions of all tiles registered to the slide, and updates self._score
        and self._pred accordingly

        """
        self._score = average_fn([tile.score for tile in self.tiles if tile.score is not None])
        self._pred = apply_decision(self._score)


class TrainingSlide(Slide):
    """
    A class for a slide that was subdivided into tiles.

    :param name: (str) An identifier for the slide.
    :param original_idx: (int) An index identifying the slide.
    :param target: (int) Label of the slide.
    :param is_cuda: (bool) Whether slide's tensor attributes will be returned as cuda tensors
    """

    def __init__(self, name, original_idx, target, tiles, is_cuda=True):
        super(TrainingSlide, self).__init__(tiles, is_cuda)
        self._name = name
        self._original_idx = original_idx
        self._target = target

    @property
    def name(self):
        """
        (str) Description identifying the slide.
        """
        return self._name

    @property
    def original_idx(self):
        """
        (int) Index identifying the slide.
        """
        return self._original_idx

    @property
    def target(self):
        """
        (int) Label of the slide.
        """
        return self._target

