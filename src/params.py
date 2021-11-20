#!/usr/bin/env python
from tensor2tensor.utils.hparam import HParams as hp


def get_params():
    return hp(modelID='180609113138',
              num_gpus=1,
              optimizer='Adam',
              loss_func='binary_crossentropy',
              activation_func='elu',  # https://keras.io/activations/
              initialization='glorot_uniform',  # Initialization of layers
              use_batch_norm=True,
              dropout_on_last_layer_only=True,
              early_stopping=False,  # Use early stopping in optimizer
              reduce_lr=False,  # Reduce learning rate during training
              save_best_only=False,
              # Save only best step in each training epoch
              use_ensemble_learning=False,  # Not implemented at the moment
              ensemble_method='Bagging',
              learning_rate=1e-4,
              dropout=0.2,  # Must be written as float for parser to work
              L1reg=0.,  # Must be written as float for parser to work
              L2reg=1e-4,  # Must be written as float for parser to work
              L1L2reg=0.,  # Must be written as float for parser to work
              decay=0.,  # Must be written as float for parser to work
              batch_norm_momentum=0.7,  # Momentum in batch normalization layers
              threshold=0.5045,  # Threshold to create binary cloud mask 511-1234, 5045 - 123, 5047 - 13
              patch_size=64,
              # Width and height of the patches the img is divided into
              overlap=20,
              # Overlap in pixels when predicting (to avoid border effects)
              overlap_train_set=0,
              # Overlap in training data patches (must be even)
              batch_size=40,
              steps_per_epoch=None,  # = batches per epoch
              epochs=5,
              norm_method='enhance_contrast',
              norm_threshold=65535,  # Threshold for the contrast enhancement
              collapse_cls=True,  # Collapse classes to one binary mask (False => multi_cls model)
              affine_transformation=True,  # Regular data augmentation
              brightness_augmentation=False,  # Experimental data augmentation
              # TODO: IF YOU CHOOSE BAND 8, IT DOES NOT MATCH THE .npy TRAINING DATA
              bands=[1, 2, 3],  # Band 8 is the panchromatic band
              # Get absolute path of the project (https://stackoverflow.com/questions/50499
              # /how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executing)
              # project_path=os.path.dirname(os.path.abspath(inspect.stack()[-1][1])) + "/")
              project_path="C:/Users/Dewire/Documents/RS-Net/",
              satellite='Landsat8',
              train_dataset='KTH_gt',
              # Training dataset (gt/fmask/sen2cor)
              test_dataset='KTH_gt',  # Test dataset (gt/fmask/sen2cor)
              split_dataset=True,  # Not used at the moment.
              test_tiles='Biome_gt')  # Used for testing if dataset is split
