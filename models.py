"""
Copyright (c) 2022- Panakeia Technologies Limited
Software licensed under GNU General Public License (GPL) version 3

Module for Pytorch encoder-decoder model implementation.
"""

import os
import numpy as np
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_DECODER_DEPTH = 5
DECODER_SCALE_FACTOR = 2


class BackboneModel(nn.Module):
    """
    A pytorch model used as backbone for an autoencoder.
    """
    def __init__(self, backbone='resnet34', out_dim=2, pretrained=True, is_cuda=True):
        super(BackboneModel, self).__init__()

        if backbone != 'resnet34':
            raise ValueError('Only resnet34 architecture is supported.')

        self.backbone = backbone
        self.model = getattr(models, self.backbone)(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_dim)

        self.is_cuda = is_cuda
        if self.is_cuda:  
            self.model.cuda()

    def forward(self, input_tensor):
        """
        Forward function.

        :param input_tensor: (torch.Tensor) input for pytorch model
        :return: (torch.Tensor) model output
        """
        input_tensor = input_tensor.cuda() if self.is_cuda else input_tensor
        return self.model(input_tensor)  # un-normalized scores for each class
    

class Decoder(nn.Module):
    """
    Custom convolutional decoder model that applies a series of upsampling, deconvolution, and non-linearity layers to
    input. A decoder currently has a depth of 5 where the first layer's output is adjusted so that the final output
    size matches that of the input image.

    :param base_in_channels: (int) number of input channels (must be visible by 2)
    :param base_out_channels: (int) number of output channels (i.e. layers)
    :param base_width: (int) width (or height) of input image used to reshape input to decoder
    :param num_final_channels: (int)
    :param kernel_size: (int) size of kernel
    :param stride: (int) controls the stride for the cross-correlation
    :param padding: (int) controls the amount of implicit zero-paddings on both sides
    """

    def __init__(
            self,
            base_in_channels=512,
            base_out_channels=256,
            base_width=7,
            num_final_channels=3,
            kernel_size=3,
            stride=1,
            padding=1
    ):
        super(Decoder, self).__init__()
        self.in_channels = base_in_channels
        self.base_width = base_width
        self.upsample = nn.Upsample(scale_factor=DECODER_SCALE_FACTOR)

        self.conv1_transpose = self._make_conv_transpose(
            base_in_channels, base_out_channels, kernel_size, stride, padding
        )
        self.conv2_transpose = self._make_conv_transpose(
            base_in_channels // 2, base_out_channels // 2, kernel_size, stride, padding
        )
        self.conv3_transpose = self._make_conv_transpose(
            base_in_channels // 4, base_out_channels // 4, kernel_size, stride, padding
        )
        self.conv4_transpose = self._make_conv_transpose(
            base_in_channels // 8, base_out_channels // 8, kernel_size, stride, padding
        )
        self.conv5_transpose = self._make_conv_transpose(
            base_in_channels // 16, num_final_channels, kernel_size, stride, padding)

    @staticmethod
    def _make_conv_transpose(in_channels, out_channels, kernel_size, stride, padding):
        """
        Function to create a transposed convolutional, i.e. deconvolutional, layer.
        :param in_channels: (int) number of input channels
        :param out_channels: (int) number of output channels (i.e. layers)
        :param kernel_size: (int) size of kernel
        :param stride: (int) controls the stride for the cross-correlation
        :param padding: (int) controls the amount of implicit zero-paddings on both sides
        :return: (nn.ConvTranspose2d) a transposed convolutional layer
        """
        return nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, input_tensor):
        """
        Forward function.

        :param input_tensor: (torch.Tensor) input for decoder
        :return: (torch.Tensor) decoder, whose output is the same as input tensor
        """
        input_tensor = input_tensor.view(input_tensor.size(0), self.in_channels, self.base_width, self.base_width)
        input_tensor = self.upsample(input_tensor)
        input_tensor = F.relu(self.conv1_transpose(input_tensor))
        input_tensor = self.upsample(input_tensor)
        input_tensor = F.relu(self.conv2_transpose(input_tensor))
        input_tensor = self.upsample(input_tensor)
        input_tensor = F.relu(self.conv3_transpose(input_tensor))
        input_tensor = self.upsample(input_tensor)
        input_tensor = F.relu(self.conv4_transpose(input_tensor))
        input_tensor = self.upsample(input_tensor)
        output = F.relu(self.conv5_transpose(input_tensor))
        return output


class EncoderDecoderModel(nn.Module):
    """
    Custom encoder-decoder model with encoder being a pre-trained backbone. Notice that this model applies
    normalisation to the image batches as first step of the forward pass.
    """

    def __init__(self, backbone_name='resnet34', pretrained=True, is_cuda=True, num_classes=2, input_width=256):
        super(EncoderDecoderModel, self).__init__()
        self.is_cuda = is_cuda
        
        backbone_model = BackboneModel(
            backbone=backbone_name,
            pretrained=pretrained,
            is_cuda=is_cuda,
            out_dim=num_classes
        )
        
        self.encoder = self._make_encoder(backbone_model)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.classifier = backbone_model.model.fc

        decoder_base_width = self._compute_decoder_base_width(input_width)
        in_features = backbone_model.model.fc.in_features
        self.in_features = in_features

        self.bridge = nn.Linear(
            in_features=self.in_features, out_features=self.in_features * decoder_base_width * decoder_base_width)

        self.decoder = self._make_decoder(
            in_channels=self.in_features, out_channels=self.in_features // 2, base_width=decoder_base_width)

        if self.is_cuda:  
            self.cuda()

    @staticmethod
    def _make_encoder(backbone_model):
        """
        Create an encoder from a resnet-like architecture.

        :param backbone_model: (torchvision.models) a pre-defined resnet-like model
        :return: (nn.Sequential) encoder network acquired from backbone model
        """

        return nn.Sequential(*list(backbone_model.model.children())[:-2])

    @staticmethod
    def _make_decoder(in_channels, out_channels, base_width):
        """
        Create a decoder module.
        :param in_channels: (int) number of input channels
        :param out_channels: (int) number of output channels (i.e. layers)
        :param base_width: (int) width (or height) of input image used to reshape input to decoder
        """

        return Decoder(
            base_in_channels=in_channels, base_out_channels=out_channels, base_width=base_width
        )

    @staticmethod
    def _compute_decoder_base_width(input_width, decoder_depth=DEFAULT_DECODER_DEPTH):
        """
        A function that automatically computes the width (and height) of the base reconstructed image, where input
        image height and width are assumed to be equal.

        :param input_width: (int) width (or height) of input image. It must be divisible by 2 ^ DEFAULT_DECODER_DEPTH,
                without a remainder, e.g. 224, 256, 320, 512.
        :decoder_depth: (int) depth of decoder, i.e. number of upsample and deconvolutional layers in the decoder
        :return: (int) width of the base reconstructed image
        """

        def _compute_width_recursively(width, depth):
            """
            Inner function that makes it possible to compute the decoder base width recursively.
            """
            if depth == 0:
                return width
            depth -= 1
            return _compute_width_recursively(width // 2, depth)

        if input_width % (2 ** DEFAULT_DECODER_DEPTH):
            raise ValueError(f'input_width must be divisible by 2 ^ {DEFAULT_DECODER_DEPTH}')

        return _compute_width_recursively(input_width, decoder_depth)

    def forward(self, input_tensor):
        """
        Forward function.

        :param input_tensor: (torch.Tensor) input for model, of shape (batch_size, 3, tile_size, tile_size)
        :return: model_out (dict of torch.Tensor variables):
            latent_vector, output of encoder, of shape (batch_size, feature_dim),
            classifier_out, of shape (batch_size, 2),
            reconstruct_out, reconstruction of input image of shape (batch_size, 3, tile_size, tile_size)
        """
        model_out = dict.fromkeys(['latent_vector', 'classifier_out', 'reconstruct_out'])
        features = self.encoder(input_tensor)
        features = self.avgpool(features)
        model_out['latent_vector'] = torch.flatten(features, 1)
        model_out['classifier_out'] = self.classifier(model_out['latent_vector'])
        input_tensor = F.relu(self.bridge(model_out['latent_vector']))
        model_out['reconstruct_out'] = self.decoder(input_tensor)
        return model_out
    

def update_model_weights(model, path_to_weights, is_cuda=True):
    """
    Get pre-trained weights from path_to_weights and update the weights in whole model.
    :param model: (model) pytorch model
    :param path_to_weights: (str) local path to where the weights are stored
    :param is_cuda: (bool) whether cuda is available
    :return: (tuple<torch.nn.module, TemperatureScaling, tuple<float, float>>)
        - model with updated weights
        - temperature scaler with updated params, if a temperature scaler was provided as a parameter and if temperature
            scaling params have been stored in the baseline model file path provided
        - An interval (a, b) contained in [0, 1] such that model claims to be uncertain if a prediction score falls
            within this interval, if such an interval is stored in the baseline model file path provided
    """

    if is_cuda:
        model_dict = torch.load(path_to_weights)
    else:
        model_dict = torch.load(path_to_weights, map_location=torch.device('cpu'))

    # load model weights
    if model is not None:
        model.load_state_dict(model_dict['state_dict'])
        print(f'Model weights loaded from {path_to_weights}')

    return model


def get_scores_preds_targets(output, target=None, is_cuda=True):
    """
    Convert model output into scores (probabilities) and predictions and return numpy versions of each.

    :param output: (torch.Tensor) Model output
    :param target: (torch.Tensor) Ground truth targets
    :param is_cuda: (bool) whether CUDA is available
    :return y_score, y_pred, y_true: (tuple) numpy versions of scores, predictions and targets (optional)
    """

    y_score = F.softmax(output, dim=1).detach().clone()
    y_score = y_score.cpu().numpy() if is_cuda else y_score.numpy()
    y_pred = np.argmax(y_score, axis=1)

    if target is not None:
        target = target.cpu().numpy() if is_cuda else target.numpy()
        y_true = np.atleast_1d(target)
        return y_score, y_pred, y_true

    return y_score, y_pred


def save_best_model_weights(estimator, epoch, save_dir, run_name):
    """
    Saves the model weights at the given epoch and add new entry to model_epoch table in panvault
    :param estimator: (ProfilerEstimator) object used to save model
    :param epoch: (int) validation epoch number
    :param save_dir: (str) directory to save output models
    :param run_name: (str) unique name of model
    """

    os.makedirs(os.path.join(save_dir, run_name), exist_ok=True)
    local_model_path = os.path.join(save_dir, run_name, 'checkpoint_best.pth')
    estimator.save_model(local_file_path=local_model_path, epoch=epoch+1)
