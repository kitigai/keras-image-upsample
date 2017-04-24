# coding=utf-8
from __future__ import absolute_import
from dataset import ImageScaleDataSet

if __name__ == "__main__":

    # Transform the images once, then run the main code to scale images

    # Change scaling factor to increase the scaling factor
    scale_factor = 2

    # Set true_upscale to True to generate smaller training images that will then be true upscaled.
    # Leave as false to create same size input and output images
    true_upscale = False

    batch_size = 1024 / scale_factor



    ImageScaleDataSet().create_ex(
        name="train", output_dir="~/datasets", batch_size=batch_size, source_dir="input_images_hires",
        scale_factor=scale_factor, true_upscale=true_upscale)

    ImageScaleDataSet().create_ex(
        name="validation", output_dir="~/datasets", batch_size=batch_size, source_dir="val_images/set5",
        scale_factor=scale_factor, true_upscale=true_upscale)
