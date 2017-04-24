# coding=utf-8
from __future__ import absolute_import
from .base import BaseSuperResolutionModel
from keras.layers import Convolution2D, merge
from keras import optimizers
from keras.models import Model
from objectives import PSNRLoss


class ExpantionSuperResolution(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ExpantionSuperResolution, self).__init__("esrcnn", scale_factor)

        self.f1 = 9
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

    def create_model(self, height=32, width=32, channels=3, load_weights=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        init = super(ExpantionSuperResolution, self).create_model(height, width, channels, load_weights)

        x = Convolution2D(self.n1, self.f1, self.f1, activation='relu', border_mode='same', name='level1')(init)

        x1 = Convolution2D(self.n2, self.f2_1, self.f2_1, activation='relu', border_mode='same', name='lavel1_1')(x)
        x2 = Convolution2D(self.n2, self.f2_2, self.f2_2, activation='relu', border_mode='same', name='lavel1_2')(x)
        x3 = Convolution2D(self.n2, self.f2_3, self.f2_3, activation='relu', border_mode='same', name='lavel1_3')(x)

        x = merge([x1, x2, x3], mode='ave')

        out = Convolution2D(channels, self.f3, self.f3, activation='relu', border_mode='same', name='output')(x)

        model = Model(init, out)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])

        if load_weights:
            model.load_weights(self.weight_path)

        self.model = model
        return model
