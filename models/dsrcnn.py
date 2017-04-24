# coding=utf-8
from __future__ import absolute_import
from keras import optimizers
from .base import BaseSuperResolutionModel
from keras.layers import Convolution2D, merge
import keras.backend as K
from keras.models import Model
from objectives import PSNRLoss


class DenoisingAutoEncoderSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(DenoisingAutoEncoderSR, self).__init__("dsrcnn", scale_factor)

        self.n1 = 64
        self.n2 = 32

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to remove / reduce noise from upscaled images.
        """
        from keras.layers.convolutional import Deconvolution2D

        # Perform check that model input shape is divisible by 4
        init = super(DenoisingAutoEncoderSR, self).create_model(height, width, channels, load_weights)

        if K.image_dim_ordering() == "th":
            output_shape = (None, channels, width, height)
        else:
            output_shape = (None, width, height, channels)

        level1_1 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(init)
        level2_1 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(level1_1)

        level2_2 = Deconvolution2D(self.n1, 3, 3, activation='relu', output_shape=output_shape, border_mode='same')(level2_1)
        level2 = merge([level2_1, level2_2], mode='sum')

        level1_2 = Deconvolution2D(self.n1, 3, 3, activation='relu', output_shape=output_shape, border_mode='same')(level2)
        level1 = merge([level1_1, level1_2], mode='sum')

        decoded = Convolution2D(channels, 5, 5, activation='linear', border_mode='same')(level1)

        model = Model(init, decoded)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)

        self.model = model
        return model
