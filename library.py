"""
Copyrights (c) 2022 - Panakeia Technologies Limited
Software licensed under GNU General Public License (GPL) version 3

A module for loading pickled data files from the disk into memory and transforming them into a format that can be
understood by Dataset modules and models.
"""

import os
import pickle
from tqdm import tqdm
import torch

from tile import TrainingTile
from slide import TrainingSlide


class DataLibrary:    
    """
    A data class that is responsible for loading the patched images into memory and transforming pickled data into
    data structures that can be processed by the Dataset class. In a nutshell it works as a transformer between raw
    tile files and a structured dataset that can be used by a pytorch DataLoader.
    """

    def __init__(self, dataset_folder, image_ids, labels):
        """
        Reads the core_dicts of the given images and generates data and file structures needed for profiler training
        and validation based on TCGA images.

        :param dataset_folder: (str) path to tiles data
        :param image_ids: (list) names of image directories
        :param labels: (list) labels of the respective image uuids, where labels[i] corresponds to image_ids[i]
        """
        if len(image_ids) != len(labels):
            raise ValueError(f"Number of given image_ids ({len(image_ids)}) does not match number of given labels "
                             f"({len(labels)}).")

        if not os.path.exists(dataset_folder):
            raise ValueError(f"{dataset_folder} must be a valid folder.")
            
        self.dataset_folder = dataset_folder
        self.image_ids = image_ids
        self.labels = labels
        self.slides = list()
        self._generate()

    def _generate(self):
        """
        Iterates over the data folder and loads pickled core dicts for each case, and appends them to a list as Slide
        objects.
        """

        for image_uuid, target in tqdm(zip(self.image_ids, self.labels), total=len(self.image_ids)):
            # Check if path locally exists
            file_path = os.path.join(self.dataset_folder, f"{image_uuid}.pickle")
            if not os.path.exists(file_path):
                print(f'{file_path} cannot be found. Skipping.')
                continue

            slide_name = file_path.split('/')[-1].split('.')[0]
            slide_idx = len(self.slides)

            with open(file_path, 'rb') as handle:
                tile_dict = pickle.load(handle)

            # Create a new slide
            tiles = self.extract_tiles_from_tile_dict(tile_dict, image_uuid=image_uuid, image_target=target)

            slide = TrainingSlide(
                name=slide_name,
                original_idx=slide_idx,
                target=target,
                tiles=tiles,
                is_cuda=torch.cuda.is_available()
            )
            self.slides.append(slide)

    @staticmethod
    def extract_tiles_from_tile_dict(tile_dict, image_uuid, image_target):
        """
        Extracts raw tile images in the form of np arrays fore core dict, saves them in .npy files and register their
        file paths with the memory manager

        :param tile_dict: (dict) containing raw tile images as np arrays
        :param image_uuid: (str) image uuid of the slide represented by given core dict
        :param image_target: (int) target of the WSI

        :return (list<Tile>) list of Tile objects representing all the tiles from the given slide
        """

        raw_tile_imgs = [tile_dict["img_patches"][i, ...] for i in range(len(tile_dict["img_patches"]))]

        tiles = []
        for i, img in enumerate(raw_tile_imgs):
            tile = TrainingTile(
                tile_img=img,
                slide_uuid=image_uuid,
                tile_index=i,
                target=image_target,
            )
            tiles.append(tile)
        return tiles

    def get_slides(self, slide_idx_filter=None):
        """
        Returns slide objects of the data library.

        :param slide_idx_filter: (list<int>) list of slide indices get slide objects for; if None, slide objects
            for all slides are returned
        """
        if slide_idx_filter is not None:
            slides = [slide for idx, slide in enumerate(self.slides) if idx in slide_idx_filter]
        else:
            slides = self.slides
        return slides

    def get_slide_indices(self):
        """
        Returns the indices of all slides managed by the data library.
        :return: (list<int>) the list of all slide indices
        """
        return list(range(len(self.slides)))
