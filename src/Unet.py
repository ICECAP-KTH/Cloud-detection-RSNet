#!/usr/bin/env python
# Needed to set seed for random generators for making reproducible experiments
import tensorflow as tf
from numpy.random import seed
from keras import backend as K

seed(1)
tf.random.set_seed(1)

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, \
    Cropping2D, Activation
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from keras.utils.generic_utils import get_custom_objects  # To use swish activation function


def swish(x):
    return (K.sigmoid(x) * x)


class Unet(object):
    def __init__(self, params):
        # Seed for the random generators
        self.seed = 1

        n_cls = 1
        n_bands = np.size(params.bands)

        # Create the model in keras
        self.model = self.__create_inference__(n_bands, n_cls, params)  # initialize the model

    def __create_inference__(self, n_bands, n_cls, params):
        # Note about BN and dropout: https://stackoverflow.com/questions/46316687/how-to-include-batch-normalization-in-non-sequential-keras-model
        get_custom_objects().update({'swish': Activation(swish)})
        inputs = Input((params.patch_size, params.patch_size, n_bands))
        # -----------------------------------------------------------------------
        conv1 = Conv2D(32, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(inputs)
        conv1 = BatchNormalization(momentum=params.batch_norm_momentum)(conv1) if params.use_batch_norm else conv1
        conv1 = Conv2D(32, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(conv1)
        conv1 = BatchNormalization(momentum=params.batch_norm_momentum)(conv1) if params.use_batch_norm else conv1
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # -----------------------------------------------------------------------
        conv2 = Conv2D(64, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(pool1)
        conv2 = BatchNormalization(momentum=params.batch_norm_momentum)(conv2) if params.use_batch_norm else conv2
        conv2 = Conv2D(64, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(conv2)
        conv2 = BatchNormalization(momentum=params.batch_norm_momentum)(conv2) if params.use_batch_norm else conv2
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # -----------------------------------------------------------------------
        conv3 = Conv2D(128, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(pool2)
        conv3 = BatchNormalization(momentum=params.batch_norm_momentum)(conv3) if params.use_batch_norm else conv3
        conv3 = Conv2D(128, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(conv3)
        conv3 = BatchNormalization(momentum=params.batch_norm_momentum)(conv3) if params.use_batch_norm else conv3
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # -----------------------------------------------------------------------
        conv4 = Conv2D(256, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(pool3)
        conv4 = BatchNormalization(momentum=params.batch_norm_momentum)(conv4) if params.use_batch_norm else conv4
        conv4 = Conv2D(256, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(conv4)
        conv4 = BatchNormalization(momentum=params.batch_norm_momentum)(conv4) if params.use_batch_norm else conv4
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        # -----------------------------------------------------------------------
        conv5 = Conv2D(512, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(pool4)
        conv5 = BatchNormalization(momentum=params.batch_norm_momentum)(conv5) if params.use_batch_norm else conv5
        conv5 = Conv2D(512, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(conv5)
        conv5 = BatchNormalization(momentum=params.batch_norm_momentum)(conv5) if params.use_batch_norm else conv5
        # -----------------------------------------------------------------------
        up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
        conv6 = Conv2D(256, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(up6)
        conv6 = Dropout(params.dropout)(conv6) if not params.dropout_on_last_layer_only else conv6
        conv6 = Conv2D(256, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(conv6)
        conv6 = Dropout(params.dropout)(conv6) if not params.dropout_on_last_layer_only else conv6
        # -----------------------------------------------------------------------
        up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = Conv2D(128, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(up7)
        conv7 = Dropout(params.dropout)(conv7) if not params.dropout_on_last_layer_only else conv7
        conv7 = Conv2D(128, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(conv7)
        conv7 = Dropout(params.dropout)(conv7) if not params.dropout_on_last_layer_only else conv7
        # -----------------------------------------------------------------------
        up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = Conv2D(64, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(up8)
        conv8 = Dropout(params.dropout)(conv8) if not params.dropout_on_last_layer_only else conv8
        conv8 = Conv2D(64, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(conv8)
        conv8 = Dropout(params.dropout)(conv8) if not params.dropout_on_last_layer_only else conv8
        # -----------------------------------------------------------------------
        up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = Conv2D(32, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(up9)
        conv9 = Dropout(params.dropout)(conv9) if not params.dropout_on_last_layer_only else conv9
        conv9 = Conv2D(32, (3, 3), activation=params.activation_func, padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(params.L2reg))(conv9)
        conv9 = Dropout(params.dropout)(conv9)
        # -----------------------------------------------------------------------
        clip_pixels = np.int32(params.overlap / 2)  # Only used for input in Cropping2D function on next line
        crop9 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(conv9)
        # -----------------------------------------------------------------------
        conv10 = Conv2D(n_cls, (1, 1), activation='sigmoid')(crop9)
        # -----------------------------------------------------------------------
        model = Model(inputs=inputs, outputs=conv10)

        return model

    def predict(self, img, params):
        # Predict batches of patches
        patches = np.shape(img)[0]  # Total number of patches
        patch_batch_size = 128
        n_cls = 1  # Number of classes to divide : We have binary classification, so we have n_cls = 1

        # Do the prediction
        predicted = np.zeros((patches, params.patch_size - params.overlap, params.patch_size - params.overlap, n_cls))
        for i in range(0, patches, patch_batch_size):
            predicted[i:i + patch_batch_size, :, :, :] = self.model.predict(img[i:i + patch_batch_size, :, :, :])

        return predicted
