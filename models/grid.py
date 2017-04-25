# coding=utf-8
from __future__ import absolute_import
from .srcnn import ImageSuperResolutionModel
import itertools
import json
from keras.callbacks import History
from keras.optimizers import Adam, Nadam, RMSprop, Adamax, Adadelta, Adagrad, SGD
from tabulate import tabulate

def create_srcnn(**kwargs):
    model = ImageSuperResolutionModel(scale_factor=2, **kwargs)
    model.create_model()
    return model


class GridExperiment(object):
    pass


class GridSearch(object):
    def __init__(self, build_fn, nb_epochs, train_data_set, validation_data_set, grid_params, search_metric, batch_sizes):
        self.build_fn = build_fn
        self.nb_epochs = nb_epochs
        self.grid_params = grid_params
        self.train_data_set = train_data_set
        self.validation_data_set = validation_data_set
        self.search_metric = search_metric
        self.batch_sizes = batch_sizes

    # train_dataset = ImageScaleDataSet.load("~/datasets/image-x2-patch32-train_small-tf")
    # validation_dataset = ImageScaleDataSet.load("~/datasets/image-x2-patch32-validation_small-tf")

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
                kwargs = dict(zip(keys, experiment_values))
                print("Execute experiment: {}/{}".format(experiment_num, len(search_values) * len(self.batch_sizes)))
                print("Experiment params: batch_size:{}, {}".format(batch_size, kwargs))
                model = self.build_fn(**kwargs)
                history = model.fit(
                    nb_epochs=self.nb_epochs, train_dataset=self.train_data_set,
                    validation_dataset=self.validation_data_set, verbose=1, batch_size=batch_size)

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
                print("Best experiment params: {}".format(json.dumps(rows[best_experiment_num], indent=4)))

                experiment_num += 1


        print(tabulate(tabular_data=rows, headers='keys', floatfmt='grid', disable_numparse=True))

def experiment(train_data_set, validation_data_set):

    gs = GridSearch(build_fn=create_srcnn, train_data_set=train_data_set, validation_data_set=validation_data_set,
                    grid_params={
                        # 'f1': [9],
                        # 'f2': [1],
                        # 'f3': [5],
                        # 'n1': [32, 64],
                        # 'n2': [32, 64, 128],
                        # 'optimizer':

                    }, nb_epochs=250, search_metric='val_PSNRLoss',
                    batch_sizes=[2, 8, 32, 64])
    gs.run()
