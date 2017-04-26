# coding=utf-8
from __future__ import absolute_import
from .base import BaseSuperResolutionModel
from keras.layers import Convolution2D, Dropout
from keras.models import Model
from keras.optimizers import Adam, RMSprop, Adagrad, SGD
from keras.objectives import mean_squared_error, mean_squared_logarithmic_error, mean_absolute_error
from objectives import PSNRLoss
import numpy as np


class ImageSuperResolutionModel(BaseSuperResolutionModel):

    def __init__(self, scale_factor, f1=9, f2=1, f3=5, n1=64, n2=32, optimizer=SGD, lr=1e-3, loss=mean_squared_logarithmic_error, dropout=None):
        super(ImageSuperResolutionModel, self).__init__("srcnn", scale_factor)

        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

        self.n1 = n1
        self.n2 = n2
        self.optimizer = optimizer
        self.lr = lr
        self.loss = loss
        self.dropout = dropout

    def create_model(self, height=None, width=None, load_weights=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        height, width = self.get_hw(height=height, width=width)
        init = super(ImageSuperResolutionModel, self).create_model(
            height=height, width=width, load_weights=load_weights)

        x = Convolution2D(filters=self.n1, kernel_size=(self.f1, self.f1), activation='relu', name='level1', padding='same')(init)
        x = Convolution2D(filters=self.n2, kernel_size=(self.f2, self.f2), activation='relu', name='level2', padding='same')(x)

        out = Convolution2D(self.channels, kernel_size=(self.f3, self.f3), name='output', padding='same')(x)

        model = Model(init, out)
        model.summary()

        model.compile(optimizer=self.optimizer(lr=self.lr), loss=self.loss, metrics=[PSNRLoss])
        if load_weights:
            model.load_weights(self.weight_path)

        self.model = model
        return model

class ImageSuperResolutionModelDropout(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ImageSuperResolutionModelDropout, self).__init__("srcnn-dropout", scale_factor)

        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

    def create_model(self, height=None, width=None, load_weights=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        height, width = self.get_hw(height=height, width=width)
        init = super(ImageSuperResolutionModelDropout, self).create_model(
            height=height, width=width, load_weights=load_weights)

        x = Convolution2D(self.n1, self.f1, self.f1, activation='relu', border_mode='same', name='level1')(init)
        x = Convolution2D(self.n2, self.f2, self.f2, activation='relu', border_mode='same', name='level2')(x)
        x = Dropout(0.25)(x)
        out = Convolution2D(self.channels, self.f3, self.f3, border_mode='same', name='output')(x)

        model = Model(init, out)

        adam = Adam(lr=1e-3)
        model.compile(optimizer=adam, loss=mean_squared_logarithmic_error, metrics=[PSNRLoss])
        if load_weights:
            model.load_weights(self.weight_path)

        self.model = model
        return model