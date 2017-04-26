# coding=utf-8
from __future__ import absolute_import
import json
import os
import cPickle
import random
import shutil

import numpy as np
import time


class Batch(object):
    def __init__(self, data_set):
        self.data_set = data_set
        self.x = list()
        self.y = list()
        self.size = 0

    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)
        self.size += 1

    def save(self, path):
        assert len(self.x) == len(self.y)
        print("Save batch with size {}: {}".format(self.size, path))
        np.savez_compressed(path, x=self.x, y=self.y)

    def load(self, path):
        npz_file = np.load(path + ".npz")
        # print("Load batch: {}".format(npz_file))
        self.x, self.y = npz_file['x'], npz_file['y']
        assert self.x.shape[0] == self.y.shape[0]
        self.size = self.x.shape[0]

    def get(self, num):
        return self.x[num], self.y[num]

    def split(self, batch_size):
        assert self.size % batch_size == 0
        for i in range(0, self.size, batch_size):
            y = self.y[i: i + batch_size]
            x = self.x[i: i + batch_size]
            yield x, y


class SamplesGenerator(object):

    def __init__(self, data_set, source_dir):
        self.data_set = data_set
        self.source_dir = source_dir

    def generator(self):
        raise NotImplementedError()


class DataSet(object):
    BATCH_CLASS = Batch
    SAMPLES_GENERATOR_CLASS = SamplesGenerator

    def __init__(self):
        self.config = dict(batches=list())
        self.dir = None

    @property
    def batch_size(self):
        return self.config['batch_size']

    @property
    def batches(self):
        return self.config['batches']

    @property
    def samples_count(self):
        return len(self.batches) * self.batch_size

    def save_batch(self, batch, output_dir):
        batch_num = len(self.batches)
        filename = "batch_{:08d}".format(batch_num)
        path = os.path.join(output_dir, filename)
        batch.save(path)
        self.batches.append(filename)

    def load_batch(self, path):
        batch = self.BATCH_CLASS(data_set=self)
        batch.load(
            os.path.join(self.dir, path)
        )
        return batch

    def generator(self, shuffle, batch_size):
        while True:
            if shuffle:
                random.shuffle(self.batches)
                for path in self.batches:
                    batch = self.load_batch(path)
                    for x, y in batch.split(batch_size):
                        yield x, y

    @classmethod
    def load(cls, path, verbose=False):
        path = os.path.expanduser(path)
        config_path = os.path.join(path, "config.pkl")

        if verbose:
            print("Load dataset: {}".format(path))

        data_set = cls()
        data_set.dir = path

        with open(config_path) as config_file:
            data_set.config = cPickle.load(config_file)
        if verbose:
            data_set.print_config()
        return data_set

    def create(self, name, output_dir, source_dir, batch_size):
        output_dir = os.path.join(
            os.path.expanduser(output_dir),
            name
        )

        self.config['name'] = name
        self.config['batch_size'] = batch_size

        print("Dataset dir: {}".format(output_dir))

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

        samples_generator = self.SAMPLES_GENERATOR_CLASS(data_set=self, source_dir=source_dir)

        current_batch = self.BATCH_CLASS(data_set=self)

        for x, y in samples_generator.generator():
            current_batch.add(x=x, y=y)

            if current_batch.size == batch_size:
                self.save_batch(current_batch, output_dir)
                current_batch = self.BATCH_CLASS(data_set=self)

        config_path = os.path.join(output_dir, "config.pkl")

        with open(config_path, 'w') as config_file:
            cPickle.dump(self.config, config_file, cPickle.HIGHEST_PROTOCOL)

        self.print_config()

    def print_config(self):
        print(json.dumps(self.config, indent=4))
