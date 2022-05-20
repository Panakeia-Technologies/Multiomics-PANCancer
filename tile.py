"""
Copyright (c) 2022- Panakeia Technologies Limited
Software licensed under GNU General Public License (GPL) version 3

Module for representations and operations on single tiles.
"""


class Tile:
    """
    Data class encapsulating all attributes of a tile within a WSI

    :param tile_index: (int) denotes index of the tile
    """

    def __init__(self, tile_index):
        self._index = tile_index
        self._score = None
        self._pred = None

    @property
    def index(self):
        """
        (int) denotes index of the tile
        """
        return self._index

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

    def update_score(self, score):
        """
        Updates the _score attribute of the tile
        :param score: (float)
        """
        self._score = score

    def update_pred(self, pred):
        """
        Updates the _pred attribute of the tile
        :param pred: (int)
        """
        self._pred = pred

    def get_img(self): 
        """
        Returns the raw tile image as a numpy array.
        :return: (numpy array)
        """
        raise NotImplementedError


class TrainingTile(Tile):
    """
    Data class encapsulating all attributes of a tile within a WSI, which is to be used in the training pipeline
    :param tile_img: (numpy array) raw tile image
    :param slide_uuid: (str) image uuid of the slide within which the tile is found
    :param target: (int) target label of the slide within which the tile is found
    :param tile_index: (int) denotes position of the tile
    """

    def __init__(self, tile_img, slide_uuid, target, tile_index):
        super().__init__(tile_index)

        self._slide_uuid = slide_uuid               # image uuid of corresponding slide
        self._tile_img = tile_img                   # tile img
        self._target = target                       # target label

    @property
    def target(self):
        """
        (int) target label of the corresponding slide
        """
        return self._target

    @property
    def slide_uuid(self):
        """
        (str) image uuid of the corresponding slide
        """
        return self._slide_uuid

    def get_img(self):
        """
        Returns the raw tile image as a numpy array.
        :return: (numpy array)
        """
        return self._tile_img
