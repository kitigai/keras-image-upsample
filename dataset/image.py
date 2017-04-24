# coding=utf-8
from __future__ import absolute_import
from .dataset import DataSet, Batch, SamplesGenerator
import os
from scipy.misc import imsave, imread, imresize
import numpy as np
import time
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from scipy.ndimage.filters import gaussian_filter
import keras.backend as K


class ImageBatch(Batch):
    def __init__(self, data_set):
        super(ImageBatch, self).__init__(data_set=data_set)
        self.image_dim_ordering = self.data_set.image_dim_ordering

    # def save(self, path):
    #     super(ImageBatch, self).save(path)
    #     for num, img in enumerate(self.x[:10]):
    #         imsave("{}_{:08d}_X.png".format(path, num), img * 255)
    #
    #     for num, img in enumerate(self.y[:10]):
    #         imsave("{}_{:08d}_y.png".format(path, num), img * 255)

    def add(self, x, y):
        """
        Add samples to batch
        :param x: x
        :type x: np.array
        :param y: y
        :type y: np.array
        :return: None
        """

        if self.image_dim_ordering == 'th':
            x = x.transpose((2, 0, 1))
            y = y.transpose((2, 0, 1))

        x = x.astype(np.float32) / 255.0
        y = y.astype(np.float32) / 255.0

        super(ImageBatch, self).add(x=x, y=y)


class ImageScaleSamplesGenerator(SamplesGenerator):
    image_scale_multiplier = 1
    img_size = 256
    # interp = 'bicubic'
    interp = 'nearest'
    gaussian_filter = False

    def __init__(self, data_set, source_dir):
        super(ImageScaleSamplesGenerator, self).__init__(data_set=data_set, source_dir=source_dir)
        self.stride = data_set.stride
        self.patch_size = data_set.patch_size
        self.scale_factor = data_set.scale_factor
        self.true_upscale = data_set.true_upscale

    def sub_image_generator(self, img):
        height, width = img.shape[:2]
        for x in range(0, height - self.patch_size, self.stride):
            for y in range(0, width - self.patch_size, self.stride):
                sub_image = img[x: x + self.patch_size, y: y + self.patch_size, :]
                yield sub_image

    def generator(self):
        lr_patch_size = self.patch_size / self.scale_factor
        hr_patch_size = self.patch_size

        for file_name in os.listdir(self.source_dir):
            image_path = os.path.join(self.source_dir, file_name)
            if not os.path.isfile(image_path):
                continue

            print("Sampling {}".format(image_path))

            img = imread(image_path, mode='RGB')

            for sample in self.sub_image_generator(img):
                y = sample

                if self.gaussian_filter:
                    # Apply Gaussian Blur to Y
                    x = gaussian_filter(y, sigma=0.5)
                else:
                    x = y

                # Subsample by scaling factor to Y
                x = imresize(x, (lr_patch_size, lr_patch_size), interp=self.interp)

                if not self.true_upscale:
                    # Upscale by scaling factor to Y
                    x = imresize(x, (hr_patch_size, hr_patch_size), interp=self.interp)

                yield x, y


class ImageScaleDataSet(DataSet):
    BATCH_CLASS = ImageBatch
    SAMPLES_GENERATOR_CLASS = ImageScaleSamplesGenerator

    @property
    def true_upscale(self):
        return self.config['true_upscale']

    @property
    def image_dim_ordering(self):
        return self.config['image_dim_ordering']

    @property
    def patch_size(self):
        return self.config['patch_size']

    @property
    def stride(self):
        return self.config['stride']

    @property
    def scale_factor(self):
        return self.config['scale_factor']

    def create_ex(self, name, output_dir, source_dir, batch_size, scale_factor,
                  true_upscale, image_dim_ordering=K.image_dim_ordering()):
        patch_size = 16 * scale_factor
        name = "image-x{scale}-patch{patch}-{name}-{image_dim_ordering}".format(
            name=name, patch=patch_size, scale=scale_factor, image_dim_ordering=image_dim_ordering)
        self.config['scale_factor'] = scale_factor
        self.config['patch_size'] = patch_size
        self.config['stride'] = patch_size / scale_factor
        self.config['true_upscale'] = true_upscale
        self.config['image_dim_ordering'] = image_dim_ordering

        super(ImageScaleDataSet, self).create(
            name=name, output_dir=output_dir, source_dir=source_dir, batch_size=batch_size)
