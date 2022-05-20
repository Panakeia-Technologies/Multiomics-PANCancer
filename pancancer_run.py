"""
Copyright (c) 2022- Panakeia Technologies Limited
Software licensed under GNU General Public License (GPL) version 3

A script to start a training job.
"""

import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")

from estimator import EncoderDecoderEstimator
from dataset import TileDataset, get_tile_dataloader
from library import DataLibrary
from transforms import get_transformation
from models import EncoderDecoderModel, update_model_weights, save_best_model_weights
from eval import evalaute_results
from utils import set_reproducibility_seed, DEFAULT_FOLD_LIST, parse_profile_file, save_results_as_json


def run_training(args, model, is_cuda=True):
    """
    Run training.

    :param args: (argparse.Namespace) Parser object used to get required variables for validation to run.
    :param model: (EncoderDecoderModel) Model object.
    :param is_cuda: (bool) Indicates cuda availability.
    """

    # Read and parse biomarker profile file
    targets_lookup, image_to_patient_lookup = parse_profile_file(args.profile_file_path)

    # Create data libraries for training and validation
    print(f'Patches are being loaded from {args.input_dir} to memory.')
    train_data_library = DataLibrary(
        dataset_folder=args.input_dir,
        image_ids=targets_lookup['Training']['image_ids'],
        labels=targets_lookup['Training']['targets'],
    )

    val_data_library = DataLibrary(
        dataset_folder=args.input_dir,
        image_ids=targets_lookup['Validation']['image_ids'],
        labels=targets_lookup['Validation']['targets'],
    )

    # Transformation
    transform = get_transformation(is_cuda=is_cuda)

    # Tile datasets
    train_tile_dataset = TileDataset(
        data_library=train_data_library,
        transform=transform,
        num_tiles_per_slide=args.num_tiles,
        replace=args.replace
    )

    val_tile_dataset = TileDataset(
        data_library=val_data_library,
        transform=transform
    )

    train_loader = get_tile_dataloader(
        mode='train', dataset=train_tile_dataset, batch_size=args.batch_size,
        weighted_sampling=True, drop_last=True
    )
    val_loader = get_tile_dataloader(mode='val', dataset=val_tile_dataset, batch_size=args.batch_size)

    # Set criteria objects
    cls_criterion = nn.CrossEntropyLoss().cuda() if is_cuda else nn.CrossEntropyLoss()
    reconstruction_criterion = nn.MSELoss()

    # Set optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Set estimator
    estimator = EncoderDecoderEstimator(
        model=model, is_cuda=is_cuda,
        tile_datasets={'train': train_tile_dataset, 'val': val_tile_dataset},
        pred_batch_size=args.batch_size, optimizer=optimizer, cls_criterion=cls_criterion,
        reconst_criterion=reconstruction_criterion, alpha=args.alpha, beta=args.beta
    )

    # Run training loop and validate every
    print('Training loop has started. Validation will be done after every epoch.')
    best_target_metric, best_validation_metrics = 0, {}
    for epoch in range(args.num_epochs):
        # Training
        print(f"-> Training: Epoch {epoch + 1}")
        results = estimator.train(train_loader)
        train_metrics = evalaute_results(results=results, image_to_patient_lookup=image_to_patient_lookup)
        save_results_as_json(train_metrics, args.results_dir, args.run_name, f'train_metrics_epoch_{epoch}')

        # Run validation after each epoch
        print(f"-> Validation: Epoch {epoch + 1}")
        results = estimator.validate(val_loader)
        val_metrics = evalaute_results(results=results, image_to_patient_lookup=image_to_patient_lookup)
        save_results_as_json(val_metrics, args.results_dir, args.run_name, f'val_metrics_epoch_{epoch}')

        # save model
        if best_target_metric <= val_metrics['image_level'][args.target_metric]:
            print(f"Validation {args.target_metric} {val_metrics['image_level'][args.target_metric]} > "
                  f"best {args.target_metric}")

            print('Saving new best model')
            save_best_model_weights(estimator, epoch, args.results_dir, args.run_name)
            best_validation_metrics = val_metrics

    # Save best val metrics
    save_results_as_json(best_validation_metrics, args.results_dir, args.run_name, 'best_validation_metrics')
    print(f'Final validation metrics from best epoch\n{best_validation_metrics}')


def run_validation(args, model, is_cuda=True):
    """
    Run validation and store results.

    :param args: (argparse.Namespace) Parser object used to get required variables for validation to run.
    :param model: (EncoderDecoderModel) Model object.
    :param is_cuda: (bool) Indicates cuda availability.
    """

    # Read and parse biomarker profile file
    targets_lookup, image_to_patient_lookup = parse_profile_file(args.profile_file_path)

    val_data_library = DataLibrary(
        dataset_folder=args.input_dir,
        image_ids=targets_lookup['Validation']['image_ids'],
        labels=targets_lookup['Validation']['targets'],
    )

    # Transformation
    transform = get_transformation(is_cuda=is_cuda)

    # Run validation in sequential mode, meaning that model will process each image individually in a for loop.
    slide_indices = val_data_library.get_slide_indices()
    joint_results = None
    print('Validation loop has started. ')
    for idx, slide_idx in enumerate(slide_indices):
        print(f"Run prediction for validation set slide {idx + 1}/{len(slide_indices)}")
        tile_dataset = TileDataset(
            data_library=val_data_library,
            transform=transform,
            slide_idx_filter=[slide_idx]
        )

        # Data loader
        tile_dataloader = get_tile_dataloader(mode='val', dataset=tile_dataset, batch_size=args.batch_size)

        # Estimator to run validation
        estimator = EncoderDecoderEstimator(
            model=model, is_cuda=is_cuda, tile_datasets={'val': tile_dataset},
            cls_criterion=torch.nn.CrossEntropyLoss().cuda() if is_cuda else nn.CrossEntropyLoss(),
            reconst_criterion=torch.nn.MSELoss().cuda() if is_cuda else nn.MSELoss()
        )

        # Validation results for slide: {idx + 1}/{len(slide_indices)}"
        results = estimator.validate(tile_dataloader)

        # Aggregate results data
        if joint_results is None:
            joint_results = results
        else:
            joint_results.merge([results])

    # Evaluate all images collectively
    test_metrics = evalaute_results(results=joint_results, image_to_patient_lookup=image_to_patient_lookup)
    save_results_as_json(test_metrics, args.results_dir, args.run_name + '_validate', 'test_metrics')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone-name', action='store', type=str, choices=['resnet34'], default='resnet34',
                        help='Specifies the backbone neural network that will be used as the encoder model. Only '
                             'resnet34 is supported.')
    parser.add_argument('--baseline-model-path', action='store', type=str, default=None, required=False,
                        help='Specifies the local path to the pth model file if transfer learning to be performed.')
    parser.add_argument('--profile-file-path', action='store', type=str, default=None, required=False,
                        help='Specifies the local path to the biomarker profile file that will be used to load image'
                             ' names, targets, and other relevant information.')
    parser.add_argument('--input-dir', action='store', type=str, default='./input', required=True,
                        help='Specifies the directory from which the input data (e.g. patches) will be loaded.')
    parser.add_argument('--results-dir', action='store', type=str, default='./results', required=True,
                        help='Specifies the directory to store the output data (e.g. results, model wrights).')
    parser.add_argument('--input-width', action='store', type=int, default=256,
                        help='Width of a dxd patch (tile) that will be fed into the model. (default: 256)')
    parser.add_argument('--gpu', action='store', type=int, default=0,
                        help='Selected GPU for training/validation. (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='Training seed to use for reproducibility. (default: 42)')
    parser.add_argument('--val-fold', type=int, default=0, choices=DEFAULT_FOLD_LIST,
                        help='Validation fold. Can be one of 0, 1, 2. The remaining folds will be used for '
                             'training (default: 0)')
    parser.add_argument('--num-tiles', type=int, default=200,
                        help='Number of tiles to sample from each image during training. If None all tiles are used. '
                             '(default: 200)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size to use during training. (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs tp train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate. (default: 0.0001)')
    parser.add_argument('--alpha', default=0.5, type=float, help='Classifier loss weight (note: setting alpha 0 '
                                                                 'disables classification, default: 0.5')
    parser.add_argument('--beta', default=1.0, type=float, help='Reconstruction loss weight (note: setting beta 0 '
                                                                'disables reconstruction, default: 1.0')
    parser.add_argument('--replace', action='store_true', default=False,
                        help='Determines if replacement to be performed during sampling of tiles. Set to True during '
                             'training')
    parser.add_argument('--target-metric', action='store', type=str, default='auc',
                        help='Target metric to use for assessing the best epoch (default: auc).')
    parser.add_argument('--run-name', action='store', type=str, default='test_run',
                        help='Unique name of the run to be used for storing results and model weights in results_dir.')
    parser.add_argument('--validate-only', action='store_true', default=False,
                        help='Activates the validation mode where no training takes place. User has to provide '
                             '--baseline-model-path.')

    # Parse arguments
    main_args = parser.parse_args()

    # Set GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(main_args.gpu)

    # Check cuda availability
    is_cuda = torch.cuda.is_available()

    # Set training seed
    set_reproducibility_seed(main_args.seed)

    # Define model
    encoder_decoder_model = EncoderDecoderModel(
        backbone_name=main_args.backbone_name,
        pretrained=True,
        is_cuda=is_cuda,
        num_classes=2,
        input_width=main_args.input_width
    )
    print(f"EncoderDecoderModel created with backbone {main_args.backbone_name}")

    if main_args.baseline_model_path is not None:
        encoder_decoder_model = update_model_weights(
            path_to_weights=main_args.baseline_model_path, model=encoder_decoder_model)

    # Run in validation mode
    if main_args.validate_only:
        run_validation(main_args, model=encoder_decoder_model, is_cuda=is_cuda)
    # Run in training mode
    else:
        run_training(main_args, model=encoder_decoder_model, is_cuda=is_cuda)
