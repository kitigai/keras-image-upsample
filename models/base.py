# coding=utf-8
from __future__ import print_function, division, absolute_import

from keras.models import Model, Sequential
from keras.layers import merge, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras.utils.np_utils import to_categorical
import keras.callbacks as callbacks
import keras.optimizers as optimizers

from advanced import HistoryCheckpoint, SubPixelUpscaling

import numpy as np
import os
import time
import img_utils


class BaseSuperResolutionModel(object):
    channels = 3

    def __init__(self, model_name, scale_factor):
        """
        Base model to provide a standard interface of adding Super Resolution models
        """
        self.model = None # type: Model
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.weight_path = os.path.join("weights/{}_x{}.h5".format(self.model, self.scale_factor))

        self.type_scale_type = "norm" # Default = "norm" = 1. / 255
        self.type_requires_divisible_shape = False
        self.type_true_upscaling = False

        self.evaluation_func = None
        self.uses_learning_phase = False

    def get_hw(self, height, width):
        height = 16 * self.scale_factor if height is None else height
        width = 16 * self.scale_factor if width is None else width
        return height, width

    def create_model(self, height, width, load_weights=False):
        """
        Subclass dependent implementation.
        """
        # if self.type_requires_divisible_shape:
        #     assert height * img_utils._image_scale_multiplier % 4 == 0, "Height of the image must be divisible by 4"
        #     assert width * img_utils._image_scale_multiplier % 4 == 0, "Width of the image must be divisible by 4"
        #
        np.random.seed(2727)



        if K.image_dim_ordering() == "th":
            shape = (self.channels, width, height)
        else:
            shape = (width, height, self.channels)

        init = Input(shape=shape)

        return init

    def fit(self, train_dataset, validation_dataset, nb_epochs, verbose):
        """
        Standard method to train any of the models.
        """

        history_fn = "{}-x{}-history.json".format(self.model_name, self.scale_factor)

        if self.model is None:
            self.create_model()

        callback_list = [
            callbacks.ModelCheckpoint(self.weight_path, monitor='val_PSNRLoss', save_best_only=True,
                                      mode='max', save_weights_only=True),
            HistoryCheckpoint(history_fn)
        ]

        print("Training model : {}".format(self.model_name))

        history = self.model.fit_generator(
            generator=train_dataset.generator(shuffle=True),
            nb_epoch=nb_epochs,
            callbacks=callback_list,
            samples_per_epoch=train_dataset.samples_count,
            validation_data=validation_dataset.generator(shuffle=True),
            nb_val_samples=validation_dataset.samples_count,
            verbose=verbose
        )

        return history

    # def evaluate(self, validation_dir):
    #     if self.type_requires_divisible_shape:
    #         _evaluate_denoise(self, validation_dir)
    #     else:
    #         _evaluate(self, validation_dir)

    def upscale(self, img_path, save_intermediate=False, return_image=False, suffix="scaled",
                patch_size=8, mode="patch", verbose=True):
        """
        Standard method to upscale an image.

        :param img_path:  path to the image
        :param save_intermediate: saves the intermediate upscaled image (bilinear upscale)
        :param return_image: returns a image of shape (height, width, channels).
        :param suffix: suffix of upscaled image
        :param patch_size: size of each patch grid
        :param verbose: whether to print messages
        :param mode: mode of upscaling. Can be "patch" or "fast"
        """
        import os
        from scipy.misc import imread, imresize, imsave

        # Destination path
        path = os.path.splitext(img_path)
        filename = path[0] + "_" + suffix + "(%dx)" % (self.scale_factor) + path[1]

        # Read image
        scale_factor = int(self.scale_factor)
        true_img = imread(img_path, mode='RGB')
        init_width, init_height = true_img.shape[0], true_img.shape[1]
        if verbose: print("Old Size : ", true_img.shape)
        if verbose: print("New Size : (%d, %d, 3)" % (init_height * scale_factor, init_width * scale_factor))

        img_height, img_width = 0, 0

        if mode == "patch" and self.type_true_upscaling:
            # Overriding mode for True Upscaling models
            mode = 'fast'
            print("Patch mode does not work with True Upscaling models yet. Defaulting to mode='fast'")

        if mode == 'patch':
            # Create patches
            if self.type_requires_divisible_shape:
                if patch_size % 4 != 0:
                    print("Deep Denoise requires patch size which is multiple of 4.\nSetting patch_size = 8.")
                    patch_size = 8

            images = img_utils.make_patches(true_img, scale_factor, patch_size, verbose)

            nb_images = images.shape[0]
            img_width, img_height = images.shape[1], images.shape[2]
            print("Number of patches = %d, Patch Shape = (%d, %d)" % (nb_images, img_height, img_width))
        else:
            # Use full image for super resolution
            img_width, img_height = self.__match_autoencoder_size(img_height, img_width, init_height,
                                                                  init_width, scale_factor)

            images = imresize(true_img, (img_width, img_height))
            images = np.expand_dims(images, axis=0)
            print("Image is reshaped to : (%d, %d, %d)" % (images.shape[1], images.shape[2], images.shape[3]))

        # Save intermediate bilinear scaled image is needed for comparison.
        intermediate_img = None
        if save_intermediate:
            if verbose: print("Saving intermediate image.")
            fn = path[0] + "_intermediate_" + path[1]
            intermediate_img = imresize(true_img, (init_width * scale_factor, init_height * scale_factor))
            imsave(fn, intermediate_img)

        # Transpose and Process images
        if K.image_dim_ordering() == "th":
            img_conv = images.transpose((0, 3, 1, 2)).astype(np.float32) / 255.
        else:
            img_conv = images.astype(np.float32) / 255.

        model = self.create_model(img_height, img_width, load_weights=True)
        if verbose: print("Model loaded.")

        # Create prediction for image patches
        result = model.predict(img_conv, batch_size=128, verbose=verbose)

        if verbose: print("De-processing images.")

         # Deprocess patches
        if K.image_dim_ordering() == "th":
            result = result.transpose((0, 2, 3, 1)).astype(np.float32) * 255.
        else:
            result = result.astype(np.float32) * 255.

        # Output shape is (original_width * scale, original_height * scale, nb_channels)
        if mode == 'patch':
            out_shape = (init_width * scale_factor, init_height * scale_factor, 3)
            result = img_utils.combine_patches(result, out_shape, scale_factor)
        else:
            result = result[0, :, :, :] # Access the 3 Dimensional image vector

        result = np.clip(result, 0, 255).astype('uint8')

        if verbose: print("\nCompleted De-processing image.")

        if return_image:
            # Return the image without saving. Useful for testing images.
            return result

        if verbose: print("Saving image.")
        imsave(filename, result)

    def __match_autoencoder_size(self, img_height, img_width, init_height, init_width, scale_factor):
        if self.type_requires_divisible_shape:
            if not self.type_true_upscaling:
                # AE model but not true upsampling
                if ((init_height * scale_factor) % 4 != 0) or ((init_width * scale_factor) % 4 != 0) or \
                        (init_height % 2 != 0) or (init_width % 2 != 0):

                    print("AE models requires image size which is multiple of 4.")
                    img_height = ((init_height * scale_factor) // 4) * 4
                    img_width = ((init_width * scale_factor) // 4) * 4

                else:
                    # No change required
                    img_height, img_width = init_height * scale_factor, init_width * scale_factor
            else:
                # AE model and true upsampling
                if ((init_height) % 4 != 0) or ((init_width) % 4 != 0) or \
                        (init_height % 2 != 0) or (init_width % 2 != 0):

                    print("AE models requires image size which is multiple of 4.")
                    img_height = ((init_height) // 4) * 4
                    img_width = ((init_width) // 4) * 4

                else:
                    # No change required
                    img_height, img_width = init_height, init_width
        else:
            # Not AE but true upsampling
            if self.type_true_upscaling:
                img_height, img_width = init_height, init_width
            else:
                # Not AE and not true upsampling
                img_height, img_width = init_height * scale_factor, init_width * scale_factor

        return img_height, img_width

