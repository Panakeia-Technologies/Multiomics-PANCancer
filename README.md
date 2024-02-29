# A systematic pan-cancer study on deep learning-based prediction of multi-omic biomarkers from routine pathology images

This repo contains the codebase to train and validate an encoder-decoder model for predicting the status of
 a multi-omic biomarker. It has been used to train and validate the 12,093 DL models predicting 4,031
 multi-omic biomarkers across 32 cancer types in our multi-omic pan-cancer study. The codebase allows running 
 end-to-end training and validation for a single biomarker as long as the required input files are provided.

## Pre-requisites
- Ubuntu: 20.04.3 LTS
- Python: 3.6.7
- CUDA Version: 11.4
- GPU: NVIDIA RTX A6000
- It is highly recommended that you run training within a docker container or a python environment with
at least one GPU. The version of the libraries installed within our local test environment (acquired via 
`pip freeze`) are given in the `version_details` file.

## Running the code
#### Data preparation and input structure:
-  The pipeline is designed to operate on tiles, each is associated with a whole slide image (WSI). TCGA images used in the study
 are available at https://portal.gdc.cancer.gov/. 
- As a pre-requisite, each WSI in the input dataset should be subdivided into patches (tiles).
- Each tile should be of type `numpy.darray` and with a shape of (d x d x 3), where d is the width and 
height of a patch and 3 is the number of channels.  
- All tiles of an image should be appended into a list, which should then be packed into a python dict, where
 the key is set to `"img_patches"` as in `{"img_patches": [numpy.darray]}`.
- Each tile dict should be stored as a pickled file. A pickled tile file's name should match the name of the
corresponding image. For instance, if the image name is `TCGA-EL-A4JZ-01Z-00-DX1`, the pickled tile file should be saved 
as `TCGA-EL-A4JZ-01Z-00-DX1.pickle`.
- Patches of each image (i.e. pickle files) should be stored in an input directory, whose path will be used to build 
internal data libraries before training/validation starts. For instance, an input directory called `inputs/` should
be structured as follows:
 
 ```
    inputs/
        ├── image_1.pickle
        ├── image_2.pickle
        └── ...
    results/
    ...
```
- In our study, we used patches of size 256 x 256 x 3, but the pipeline should support any tile size, as long
as the `--image-width` argument is a match.

#### Profile file:
- A profile file contains all the data required to train and validate a DL model for predicting the status of
 a biomarker, including name of images (slides), biomarker status associated with each image, and the folds each image
 has been assigned to. In addition, it should also contain the patient ids associated with each image, so that the
 evaluation metrics can be computed at the patient level. 
 - The pipeline expects the user to provide a profile file in a CSV format containing the following 4 columns: 
image_id, target, cv_fold, patient_id. 
    - `image_id`: A unique identifier for each image (slide) in the dataset. 
    - `patient_id`: Indicates the patient associated with an image. One patient can have multiple images.
    - `target`: Indicates the ground-truth status of a biomarker. It can be either 0 or 1. 
    - `cv_fold`: Represents the cross-validation fold of an image. The current study allows to define 3 folds, i.e. 
this attribute can be of 0, 1, or 2.
- Each `image_id` should have a corresponding data file that contains the tiles and has the same name, e.g. the image 
`TCGA-EL-A4JZ-01Z-00-DX1` should have a pickle file called `TCGA-EL-A4JZ-01Z-00-DX1.pickle` in the input directory 
specified by the user.  
- We have provided the profile file used for the BRAF mutation model in thyroid carcinoma (TCGA-THCA), 
called `tcga_thca_mutation_BRAF.csv`, as a reference.
 
#### How to run code: 
- Unzip the content of the zip file to a folder.
- `cd` to that folder and run `python pancancer_run.py --help` to see the following arguments. `{}` indicates 
the valid choices an argument can get.

```
   --backbone-name {resnet34}
                        Specifies the backbone neural network that will be
                        used as the encoder model. Only resnet34 is supported.
  --baseline-model-path BASELINE_MODEL_PATH
                        Specifies the local path to the pth model file if
                        transfer learning to be performed.
  --profile-file-path PROFILE_FILE_PATH
                        Specifies the local path to the biomarker profile file
                        that will be used to load image names, targets, and
                        other relevant information.
  --input-dir INPUT_DIR
                        Specifies the directory from which the input data
                        (e.g. patches) will be loaded.
  --results-dir RESULTS_DIR
                        Specifies the directory to store the output data (e.g.
                        results, model wrights).
  --input-width INPUT_WIDTH
                        Width of a dxd patch (tile) that will be fed into the
                        model. (default: 256)
  --gpu GPU             Selected GPU for training/validation. (default: 0)
  --seed SEED           Training seed to use for reproducibility. (default:
                        42)
  --val-fold {0,1,2}    Validation fold. Can be one of 0, 1, 2. The remaining
                        folds will be used for training (default: 0)
  --num-tiles NUM_TILES
                        Number of tiles to sample from each image during
                        training. If None all tiles are used. (default: 200)
  --batch-size BATCH_SIZE
                        Batch size to use during training. (default: 128)
  --num-epochs NUM_EPOCHS
                        Number of epochs tp train (default: 10)
  --lr LR               Learning rate. (default: 0.0001)
  --alpha ALPHA         Classifier loss weight (note: setting alpha 0 disables
                        classification, default: 0.5
  --beta BETA           Reconstruction loss weight (note: setting beta
                        0 disables reconstruction, default: 1.0
  --replace             Determines if replacement to be performed during
                        sampling of tiles.
  --target-metric TARGET_METRIC
                        Target metric to use for assessing the best epoch
                        (default: auc).
  --run-name RUN_NAME   Unique name of the run to be used for storing results
                        and model weights in results_dir.
  --validate-only       Activates the validation mode where no training takes
                        place. User has to provide --baseline-model-path.
```

- After setting up the directories one can train a model with the default parameters using the command below:
```
python pancancer_run.py --val-fold 0 --profile-file-path ./tcga_thca_mutation_BRAF.csv --input-dir ./input \
--run-name test_training_model --results-dir ./results --replace --num-epochs 10
```

- The pipeline also allows running a standalone validation process where a pre-trained model can be evaluated, if
a baseline model path is provided and the `--validate` argument is passed, like in the following command:

```
python pancancer_run.py --baseline-model-path ./input/test_model.pth --val-fold 0 \
--profile-file-path /tcga_thca_mutation_BRAF.csv --input-dir ./input --run-name final_test_patient \
--results-dir ./results --validate
```

#### Run time
The run time of the code depends on various factors, such as type of GPU, input size, batch size, number of images etc.
In our test environment it took ~30 seconds to validate a model on an image with 1250 tiles.

## Copyright 
Copyright (c) 2022- Panakeia Technologies Limited

## Licence
Software licensed under GNU General Public License (GPL) version 3
