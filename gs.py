# coding=utf-8
from __future__ import absolute_import
from models.grid import experiment
from dataset import ImageScaleDataSet

train_dataset = ImageScaleDataSet.load("~/datasets/image-x2-patch32-train_grid-tf")
validation_dataset = ImageScaleDataSet.load("~/datasets/image-x2-patch32-validation_small-tf")

experiment(train_data_set=train_dataset, validation_data_set=validation_dataset)
