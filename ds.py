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

    # batch_size = 2048 #2048 / scale_factor

    ImageScaleDataSet().create_ex(
        name="train_small", output_dir="~/datasets", source_dir="input_images_lrres",
        scale_factor=scale_factor, true_upscale=true_upscale)

    # ImageScaleDataSet().create_ex(
    #     name="train_grid", output_dir="~/datasets", source_dir="grid_search_lrres",
    #     scale_factor=scale_factor, true_upscale=true_upscale)


    # ImageScaleDataSet().create_ex(
    #     name="train_lrres", output_dir="~/datasets", source_dir="input_images_hires",
    #     scale_factor=scale_factor, true_upscale=true_upscale)

    # ImageScaleDataSet().create_ex(
    #     name="validation_small", output_dir="~/datasets", source_dir="val_images/set5",
    #     scale_factor=scale_factor, true_upscale=true_upscale)
