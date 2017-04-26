# coding=utf-8
from __future__ import absolute_import
import numpy as np
np.random.seed(27)

from models.grid import experiment
from dataset import ImageScaleDataSet

# train_data_set_path = "~/datasets/image-x10-patch160-train_small-tf"
# validation_data_set_path = "~/datasets/image-x10-patch160-validation_small-tf"

train_data_set_path = "~/datasets/image-x2-patch32-train_grid-tf"
validation_data_set_path = "~/datasets/image-x2-patch32-validation_small-tf"


experiment(train_data_set_path=train_data_set_path, validation_data_set_path=validation_data_set_path)
