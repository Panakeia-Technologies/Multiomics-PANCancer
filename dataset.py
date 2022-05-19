"""
Copyrights (c) 2022 - Panakeia Technologies Limited
Software licensed under GNU General Public License (GPL) version 3

Pytorch Dataset and DataLoader implementations for tile-based profiler models.
"""

import numpy as np
import torch.utils.data as data


class TileDataset(data.Dataset):  
    """
    Dataset class for all tiles in the library that will undergo model inference.
    """
    def __init__(self, data_library, transform=None, slide_idx_filter=None, num_tiles_per_slide=None, replace=False):
        """
        :param data_library: (DataLibrary) container for all slides, tiles (patches), and targets
        :param transform: (list) defines list of transformations to be applied to each tile
        :param slide_idx_filter: (list<int>) list of slide indices from the data library to use; if None, all slides
            are used
        :param num_tiles_per_slide: (int?) number of tiles to sample from each slide. If None, all tiles will be used.
        :param replace: (bool) determines if sampling done with replacement or not.
        """

        if transform is None:
            raise ValueError('transform cannot be None')

        self.data_library = data_library
        self.transform = transform
        self.slide_idx_filter = slide_idx_filter

        self.slides = self.data_library.get_slides(slide_idx_filter)
        self.tiles = [tile for slide in self.slides for tile in slide.tiles]
        
        self.random = np.random.RandomState(42)
        self.num_tiles_per_slide = num_tiles_per_slide
        self.replace = replace

        self.tiles = self.create_tile_pool()
        self.sample_weights = self.compute_sample_weights()

    def create_tile_pool(self):
        """
        Sample m tiles from each slide where m = num_tiles_per_slide. If replacement is False and a slide
        has fewer tiles than m, then use all tiles.

        :return (list<Tile>)
        """

        tile_pool = []
        for slide in self.slides:
            if self.num_tiles_per_slide is None:
                sampled_tiles = slide.tiles
            else:
                k = min(self.num_tiles_per_slide, len(slide.tiles)) \
                    if self.replace is False else self.num_tiles_per_slide
                sampled_tiles = self.random.choice(slide.tiles, k, replace=self.replace)

            tile_pool.extend(sampled_tiles)

        return tile_pool

    def compute_sample_weights(self):
        """
        Compute sample weights with respect to the inverse frequency of class weights.

        :return (list<float>): containing probabilities of sampling for each data item.
        """

        class_counts = {
            target : len([tile for tile in self.tiles if tile.target == target])
            for target in np.unique([tile.target for tile in self.tiles])
        }
        print(f'Class counts: {class_counts}')

        class_weights = {
            target: len(self.tiles) / class_counts[target] for target in class_counts
        }

        return [class_weights[tile.target] for tile in self.tiles]

    def __getitem__(self, index):
        """
        Overriding parent class torch.utils.data.Dataset's __getitem__ function to reflect our own dataset structure.
        """
        tile = self.tiles[index]
        img = self.tiles[index].get_img()
        img = self.transform(img)
        return img, tile.target, index

    def __len__(self):
        """
        Overriding parent class torch.utils.data.Dataset's __len__ function to reflect our own dataset structure.
        """
        return len(self.tiles)
    

def get_tile_dataloader(mode, dataset, batch_size, shuffle=False, weighted_sampling=False, drop_last=False):
    """
    Gets a Pytorch data loader from a tile dataset object.

    :param mode: (str) train, val, predict
    :param dataset: (TileDataset) to pass into data loader
    :param batch_size: (int) of the data loader
    :param shuffle: (bool) turns on shuffling during data loading; only supported for mode 'train'
    :param weighted_sampling: (bool) whether to perform weighted sampling; only supported for mode 'train'
    :param drop_last: (bool) whether to drop the last batch; not possible if mode is 'predict'
    :return: (torch.utils.data.DataLoader) for segmentation based iteration
    """
    if mode == 'train':
        if weighted_sampling:
            if not hasattr(dataset, 'sample_weights'):
                raise ValueError(f"Given dataset of type {type(dataset)} does not support weighted sampling.")
            sampler = data.sampler.WeightedRandomSampler(dataset.sample_weights, num_samples=len(dataset),
                                                         replacement=True)
            data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                          drop_last=drop_last)
        else:
            data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset, batch_size=batch_size, drop_last=drop_last)
    else:
        raise ValueError("The mode argument must be one of ['train', 'val'].")
    return data_loader
