"""
Copyrights (c) 2022 - Panakeia Technologies Limited
Software licensed under GNU General Public License (GPL) version 3

Module for providing an API for training, and validation routines for encoder-decoder models.
"""

import numpy as np
import torch
from operator import itemgetter

from slide import Slide
from models import get_scores_preds_targets


class EncoderDecoderEstimator:
    """
    An Estimator class for training, validation, and feature extraction routines, specific to auto-encoder models
    for classification with multi-loss.

    :param model: (torch.nn.Module) the multi-loss encoder-decoder model used to classify tiles
    :param is_cuda: (bool) whether to use CUDA
    :param tile_datasets: (dict) of TileDataset objects 
    :param pred_batch_size: (int) batch size for the train / pred / val step after the inference step
    :param optimizer: (torch.optim.Optimizer) optimizer used in training loop; only needed for train
    :param cls_criterion: (torch.nn) loss function for classification; only needed for train and val
    :param reconst_criterion: (torch.nn) loss function for reconstruction; only needed for train and val
    :param alpha: (float) weighting multiplier for the classifier loss; only needed for train and val
    :param beta: (float) weighting multiplier for the reconstruction loss; only needed for train and val
    :param num_classes: (int) number of model output classes
    """

    def __init__(
            self, model, is_cuda, tile_datasets=None,
            pred_batch_size=None, optimizer=None, cls_criterion=None, reconst_criterion=None, 
            alpha=1.0, beta=1.0, scheduler=None, num_classes=2
    ):
        self.model = model
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.is_cuda = is_cuda
        self.tile_datasets = tile_datasets
        self.pred_batch_size = pred_batch_size
        self.cls_criterion = cls_criterion
        self.reconst_criterion = reconst_criterion
        self.alpha = alpha
        self.beta = beta

    def compute_total_loss(self, model_out, inputs, target): 
        """
        A function that computes reconstruction and classification losses
        
        :param model_out: (dict of torch.Tensors) dict of model outputs including reconstruct_out, classifier_out,
            latent_vector, logvar
        :param inputs: (torch.Tensor) tile patches as inputs
        :param target: (torch.Tensor) class labels
        :return: loss, reconstruct_loss, cls_loss
        """
        reconstruct_loss = self.reconst_criterion(model_out['reconstruct_out'], inputs)
        cls_loss = self.cls_criterion(model_out['classifier_out'], target)
        loss = self.beta * reconstruct_loss + self.alpha * cls_loss
        return loss, reconstruct_loss, cls_loss
    
    def train(self, data_loader):
        """
        Training routine that does a forward pass, updates the model wrt the loss for each batch.

        :param data_loader: (torch.utils.data.DataLoader) data loader for generating batches
        :return: (EncoderDecoderEstimatorResults) corresponding to epoch loss, model scores, model
                predictions, true labels, and dictionary for individual losses, respectively.
        """
                
        print(f'---> Training ({len(data_loader)} batches)')
        self.model.train()

        tile_level_results = EncoderDecoderEstimatorResults()
        for i, (inputs, target, access_idx) in enumerate(data_loader):
            # Send target to gpu
            target = target.cuda() if self.is_cuda else target
            target = target.long()
            self.optimizer.zero_grad()

            model_out = self.model(inputs)
            loss, reconstruct_loss, cls_loss = self.compute_total_loss(model_out, inputs, target)
            loss.backward()
            self.optimizer.step()

            # Keep track of loss, scores, preds, targets for each batch
            score, pred, true = get_scores_preds_targets(model_out['classifier_out'], target, is_cuda=self.is_cuda)
            batch_loss = (loss.item() * inputs.size(0)) / len(data_loader.dataset)
            print(f'----- Batch: {i + 1}/{len(data_loader)} -- loss: {batch_loss}')

            tile_level_results.update_targets(target, access_idx)
            tile_level_results.update_losses(reconstruct_loss.item(), cls_loss.item())
            tile_level_results.update_preds(score, pred, true, batch_loss)

        image_level_results = self.aggregate(tile_level_results, data_loader.dataset)

        return image_level_results

    @torch.no_grad()
    def validate(self, data_loader):
        """
        Validation routine that does a forward pass and returns the model results.

        :param data_loader: (torch.utils.data.DataLoader) data loader for generating batches
        :return: (EncoderDecoderEstimatorResults) corresponding to epoch loss, model scores, model
                predictions, true labels, and dictionary for individual losses, respectively.
        """
        if self.tile_datasets is None or 'val' not in self.tile_datasets:
            raise ValueError("For the 'validate' function 'tile_datasets['val']' needs to be provided.")

        print(f'---> Validation ({len(data_loader)} batches)')

        self.model.eval()
        tile_level_results = EncoderDecoderEstimatorResults()

        for i, (inputs, target, access_idx) in enumerate(data_loader):
            # Send target to gpu
            target = target.cuda() if torch.cuda.is_available() else target
            target = target.long()

            # Forward pass
            model_out = self.model(inputs)

            # Compute losses
            loss, reconstruct_loss, cls_loss = self.compute_total_loss(model_out, inputs, target)
            tile_level_results.update_losses(reconstruct_loss.item(), cls_loss.item())
            batch_loss = (loss.item() * inputs.size(0)) / len(data_loader.dataset)

            print(f'----- Batch: {i + 1}/{len(data_loader)} -- loss: {batch_loss}')

            # Keep track of loss, scores, preds, targets for each batch
            score, pred, true = get_scores_preds_targets(model_out['classifier_out'], target, is_cuda=self.is_cuda)
            tile_level_results.update_preds(score, pred, true, batch_loss)
            tile_level_results.update_targets(target, access_idx)
            
        image_level_results = self.aggregate(tile_level_results, data_loader.dataset)

        return image_level_results
    
    @staticmethod
    def aggregate(tile_level_results, tile_dataset):
        """
        Aggregates tile level results to slide level

        :param tile_level_results: (EncoderDecoderEstimatorResults) on a tile level
        :param tile_dataset: (TileClassifierDataset) keeping track of slide indices of tiles
        :return: (EncoderDecoderEstimatorResults) on slide level
        """

        # Create new ImageLevelEstimatorResults object
        slide_level_results = ImageLevelEstimatorResults()
        
        # Rearrange results using access_idx_init
        ordered_tiles = itemgetter(*tile_level_results.access_idx_init)(tile_dataset.tiles)
        tiles_by_slide = {}
        for tile, score, pred in zip(ordered_tiles, tile_level_results.y_score, tile_level_results.y_pred):
            tile.update_score(score)
            tile.update_pred(pred)
            tiles_by_slide.setdefault(tile.slide_uuid, []).append(tile)

        # Aggregate results and update slide_level_results
        slides = {slide_uuid: Slide(tiles) for slide_uuid, tiles in tiles_by_slide.items()}
        for uuid, slide in slides.items():
            if len(slide.tiles) > 0:
                slide.aggregate_tile_results()
                slide_level_results.slide_uuids.append(uuid)
                if not all(tile.target == slide.tiles[0].target for tile in slide.tiles):
                    raise ValueError('Got differing values of tile targets from the same slide')
                else:
                    target = slide.tiles[0].target
                slide_level_results.y_true.append(target)
                slide_level_results.y_score.append(slide.score[-1])
                slide_level_results.y_pred.append(slide.pred)
            else:
                raise ValueError("No tiles in slides")

        slide_level_results.add_pre_aggregation_results(tile_level_results)

        return slide_level_results
    
    def save_model(self, local_file_path, epoch=None):
        """
        Saves model checkpoint, which may include weights of model / optimizer / lr scheduler / calibrator

        :param local_file_path: (str) name of the local filepath to save model to
        :param epoch: (int?) optional. Gives current epoch of training/validation
        """
        checkpoint = dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if self.model is not None:
            checkpoint['state_dict'] = self.model.state_dict()
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(checkpoint, local_file_path)
            
            
class EstimatorResults:
    """
    An EstimatorResults object is responsible for keeping track of all the results produced by a model training,
    validation or prediction step run by an Estimator object.
    """

    def __init__(self):
        """
        Creates data structures to store results in.
        """
        self.y_score = []
        self.y_pred = []
        self.y_true = []
        self.loss = None
        self.batch_losses = []

    def update_preds(self, scores, preds, **kwargs):
        """
        Function for processing batched model predictions and associated information.

        :param scores: (list-like) batched model output scores (batch_size, ...)
        :param preds: (list-like) batched model binary predictions (batch_size, ...)
        """
        self.y_score.extend(scores)
        self.y_pred.extend(preds)

    def merge(self, results_objects):
        """
        Merges other EstimatorResults objects into this object. Assumes that all objects contain data for the same
        attributes.

        :param results_objects: (list<EstimatorResults>) other estimator results objects
        """
        for results_object in results_objects:
            self.y_score.extend(results_object.y_score)
            self.y_pred.extend(results_object.y_pred)
            self.y_true.extend(results_object.y_true)
            if self.loss is not None and results_object.loss is not None:
                self.loss += results_object.loss

    def get_results_dict(self) -> dict:
        """
        :return: (dict) summarising the obtained results
        """
        res_dict = {}
        if len(self.y_score) > 0:
            res_dict['y_score'] = self.y_score
        if len(self.y_pred) > 0:
            res_dict['y_pred'] = self.y_pred
        if len(self.y_true) > 0:
            res_dict['y_true'] = self.y_true
        if self.loss is not None:
            res_dict['loss'] = self.loss
        return res_dict


class ImageLevelEstimatorResults(EstimatorResults):
    """
    Class for storing image level results
    """

    def __init__(self):
        super().__init__()
        self.pre_aggregation_results = None
        self.slide_uuids = []

    def add_pre_aggregation_results(self, pre_aggregation_results):
        """
        Function for recording pre-aggregation results which we may want to save for E&I purposes later on
        e.g. tile-level results

        :param pre_aggregation_results: (EstimatorResults) results objects produced before aggregation
        """

        if self.pre_aggregation_results is None:
            self.pre_aggregation_results = pre_aggregation_results
        else:
            raise ValueError('Pre-aggregation results already exist in ImageLevelEstimatorResults object')

    def merge(self, results_objects):
        """
        Merges other EstimatorResults objects into this object. Assumes that all objects contain data for the same
        attributes.

        :param results_objects: (list<ImageLevelEstimatorResults>) other estimator results objects
        """
        super(ImageLevelEstimatorResults, self).merge(results_objects)
        for results_object in results_objects:
            if self.pre_aggregation_results is not None and results_object.pre_aggregation_results is not None:
                self.pre_aggregation_results.merge([results_object.pre_aggregation_results])
            self.slide_uuids.extend(results_object.slide_uuids)

    def get_results_dict(self) -> dict:
        """
        Get results_dict.
        """
        res_dict = super().get_results_dict()
        if self.pre_aggregation_results is not None:
            res_dict['pre_aggregation_results'] = self.pre_aggregation_results
        if self.slide_uuids is not None:
            res_dict['image_ids'] = self.slide_uuids
        return res_dict


class EncoderDecoderEstimatorResults(EstimatorResults):
    """
    Data class for processing the batched estimator results in train, validate, feature_extraction and predict, and
    providing them in a concise format.
    """

    def __init__(self):
        super(EncoderDecoderEstimatorResults, self).__init__()
        self.reconstruction_losses = []
        self.classification_losses = []
        self.access_idx_init = []
        self.class_labels = []
        self.slide_uuids = []

    def merge(self, results_objects):
        """
        Merges other EstimatorResults objects into this object. Assumes that all objects contain data for the same
        attributes. Notice that access_idx_init will not be merged but reset.

        :param results_objects: (list<EncoderDecoderEstimatorResults>) other estimator results objects
        """
        super(EncoderDecoderEstimatorResults, self).merge(results_objects)
        for results_object in results_objects:
            self.reconstruction_losses.extend(results_object.reconstruction_losses)
            self.classification_losses.extend(results_object.classification_losses)       
            self.class_labels.extend(results_object.class_labels)
            self.access_idx_init = np.arange(len(self.class_labels))
            if self.slide_uuids is not None:
                self.slide_uuids.extend(results_object.slide_uuids)

    def update_preds(self, scores, preds, targets=None, loss=None):
        """
        Function for processing batched encoder-decoder model predictions and associated information.

        :param scores: (list-like) batched model output scores (batch_size, ...)
        :param preds: (list-like) batched model binary predictions (batch_size, ...)
        :param targets: (list-like) batched actual binary targets (batch_size, ...); optional
        :param loss: (float) normalised loss of the batch; optional
        """
        super(EncoderDecoderEstimatorResults, self).update_preds(scores=scores, preds=preds)
        if targets is not None:
            self.y_true.extend(targets)

        if loss is not None:
            if loss == np.Inf or loss == np.NINF or loss == np.nan or loss == float("inf") or loss == float("-inf"):
                print(f'Warning: Batch loss is infinity or nan: {loss}')
                loss = np.nan_to_num(loss).item()
            self.batch_losses.append(loss)
            if self.loss is None:
                self.loss = loss
            else:
                self.loss += loss
                
    def update_targets(self, targets, access_idx):
        """
        Function for updating tile-level targets and access idx.

        :param access_idx: the original indices of the tiles of a batch
        :param targets: the targets of all the tiles of a batch
        """
        self.class_labels.extend(targets.detach().cpu().tolist())
        self.access_idx_init.extend(access_idx.detach().cpu().tolist())

    def update_losses(self, reconstruction_loss=None, classification_loss=None):
        """
        Function for processing different losses resulting from a batch prediction step.

        :param reconstruction_loss: (float) the reconstruction loss; optional
        :param classification_loss: (float) the classification loss; optional
        """
        if reconstruction_loss is not None:
            self.reconstruction_losses.append(reconstruction_loss)
        if classification_loss is not None:
            self.classification_losses.append(classification_loss)

    def get_results_dict(self):
        """
        :return: (dict) the processed results
        """
        res_dict = super().get_results_dict()
        loss_dict = {}
        if len(self.reconstruction_losses) > 0:
            loss_dict['reconstruction'] = self.reconstruction_losses
        if len(self.classification_losses) > 0:
            loss_dict['classification'] = self.classification_losses
        if len(loss_dict) > 0:
            res_dict['loss_dict'] = loss_dict
        if len(self.class_labels) > 0:
            res_dict['class_labels'] = self.class_labels
        if len(self.access_idx_init) > 0:
            res_dict['access_idx'] = self.access_idx_init
        if self.slide_uuids is not None:
            res_dict['image_ids'] = self.slide_uuids
        return res_dict
