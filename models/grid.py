# coding=utf-8
from __future__ import absolute_import
from .srcnn import ImageSuperResolutionModel
import itertools
import json
from keras.callbacks import History
from keras.optimizers import Adam, Nadam, RMSprop, Adamax, Adadelta, Adagrad, SGD
from tabulate import tabulate
from dataset import ImageScaleDataSet
import numpy as np
import random


def create_srcnn(**kwargs):
    model = ImageSuperResolutionModel(**kwargs)
    model.create_model()
    return model


class GridExperiment(object):
    pass


class GridSearch(object):
    seed = 27
    def __init__(self, build_fn, nb_epochs, train_data_set_path, validation_data_set_path, grid_params, search_metric, batch_sizes):
        self.build_fn = build_fn
        self.nb_epochs = nb_epochs
        self.grid_params = grid_params
        self.train_data_set_path = train_data_set_path
        self.validation_data_set_path = validation_data_set_path
        self.search_metric = search_metric
        self.batch_sizes = batch_sizes


    def run(self):
        keys = self.grid_params.keys()
        values = []
        for k in keys:
            values.append(
                self.grid_params[k]
            )

        search_values = list(itertools.product(*values))
        experiments = []
        best_experiment_num = None
        experiment_num = 0
        rows = []
        for batch_size in self.batch_sizes:
            for experiment_values in search_values:
                np.random.seed(self.seed)
                random.seed(self.seed)
                kwargs = dict(zip(keys, experiment_values))
                print("Execute experiment: {}/{}".format(experiment_num, len(search_values) * len(self.batch_sizes)))
                print("Experiment params: batch_size:{}, {}".format(batch_size, kwargs))
                model = self.build_fn(**kwargs)
                train_data_set = ImageScaleDataSet.load(self.train_data_set_path)
                validation_data_set = ImageScaleDataSet.load(self.validation_data_set_path)
                history = model.fit(
                    nb_epochs=self.nb_epochs, train_dataset=train_data_set,
                    validation_dataset=validation_data_set, verbose=1, batch_size=batch_size)

                best_metric_value = None
                best_epoch = 0
                for epoch in history.epoch:
                    metric_value = history.history[self.search_metric][epoch]
                    if best_metric_value is None:
                        best_metric_value = history.history[self.search_metric][epoch]
                        best_epoch = epoch
                    elif metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_epoch = epoch

                experiments.append(
                    dict(
                        kwargs=kwargs, num=experiment_num, best_metric_value=best_metric_value, history=history.history,
                        batch_size=batch_size
                    ),
                )
                experiment_row = {
                    'exp #': experiment_num,
                    self.search_metric: best_metric_value,
                    'epoch': best_epoch + 1,
                    'batch': batch_size,
                }
                experiment_row.update(kwargs)

                rows.append(experiment_row)

                # print("Experiment history: {}".format(json.dumps(history.history, indent=4)))
                # print("Experiment epoch: {}".format(json.dumps(history.epoch, indent=4)))
                print("Best value {}: {}".format(self.search_metric, best_metric_value))

                if best_experiment_num is None:
                    best_experiment_num = experiment_num
                elif best_metric_value > experiments[best_experiment_num]['best_metric_value']:
                    best_experiment_num = experiment_num

                print("Best experiment: {}".format(best_experiment_num))
                print("Best experiment params: {}".format(rows[best_experiment_num], indent=4))

                experiment_num += 1

        print(tabulate(tabular_data=rows, headers='keys', floatfmt='grid', disable_numparse=True))


def experiment(train_data_set_path, validation_data_set_path):

    gs = GridSearch(build_fn=create_srcnn,
                    train_data_set_path=train_data_set_path,
                    validation_data_set_path=validation_data_set_path,
                    grid_params={
                        'scale_factor': [2],
                        # 'f1': [9],
                        # 'f2': [1],
                        # 'f3': [5],
                        # 'n1': [64, 128],
                        # 'n2': [32, 64],
                        'optimizer': [Adadelta],
                        'lr': [0.5]

                    }, nb_epochs=500, search_metric='val_PSNRLoss',
                    batch_sizes=[8])
    gs.run()
