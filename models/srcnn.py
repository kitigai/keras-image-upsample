# coding=utf-8
from __future__ import absolute_import
from .base import BaseSuperResolutionModel
from keras.layers import Convolution2D
from keras.models import Model
from keras.optimizers import Adam
from keras.objectives import mse
from objectives import PSNRLoss


class ImageSuperResolutionModel(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ImageSuperResolutionModel, self).__init__("srcnn", scale_factor)

        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

    def create_model(self, load_weights=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        init = super(ImageSuperResolutionModel, self).create_model(load_weights=load_weights)

        x = Convolution2D(self.n1, self.f1, self.f1, activation='relu', border_mode='same', name='level1')(init)
        x = Convolution2D(self.n2, self.f2, self.f2, activation='relu', border_mode='same', name='level2')(x)
        channels = 3
        out = Convolution2D(channels, self.f3, self.f3, border_mode='same', name='output')(x)

        model = Model(init, out)

        adam = Adam(lr=1e-3)
        model.compile(optimizer=adam, loss=mse, metrics=[PSNRLoss])
        if load_weights:
            model.load_weights(self.weight_path)

        self.model = model
        return model
