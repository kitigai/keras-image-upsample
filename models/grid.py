# coding=utf-8
from __future__ import absolute_import
from .srcnn import ImageSuperResolutionModel
import itertools
import json
from keras.callbacks import History

def create_srcnn(f1, f2, f3, **kwargs):
    model = ImageSuperResolutionModel(scale_factor=2, f1=f1, f2=f2, f3=f3)
    model.create_model()
    return model


class GridExperiment(object):
    pass


class GridSearch(object):
    def __init__(self, build_fn, nb_epochs, train_data_set, validation_data_set, grid_params, search_metric):
        self.build_fn = build_fn
        self.nb_epochs = nb_epochs
        self.grid_params = grid_params
        self.train_data_set = train_data_set
        self.validation_data_set = validation_data_set
        self.search_metric = search_metric

    # train_dataset = ImageScaleDataSet.load("~/datasets/image-x2-patch32-train_small-tf")
    # validation_dataset = ImageScaleDataSet.load("~/datasets/image-x2-patch32-validation_small-tf")

    def run(self):
        keys = self.grid_params.keys()
        values = []
        for k in keys:
            values.append(
                self.grid_params[k]
            )

        search_values = itertools.product(*values)
        experiments = []
        best_experiment_num = None
        for experiment_num, experiment_values in enumerate(search_values):

            kwargs = dict(zip(keys, experiment_values))
            print("Execute experiment: {}".format(experiment_num))
            print("Experiment params: {}".format(kwargs))
            model = self.build_fn(**kwargs)
            history = model.fit(
                nb_epochs=self.nb_epochs, train_dataset=self.train_data_set,
                validation_dataset=self.validation_data_set, verbose=1)

            best_metric_value = None

            for epoch in history.epoch:
                metric_value = history.history[self.search_metric][epoch]
                if best_metric_value is None:
                    best_metric_value = history.history[self.search_metric][epoch]
                elif metric_value > best_metric_value:
                    best_metric_value = metric_value

            experiments.append(
                dict(kwargs=kwargs, history=history.history, num=experiment_num, best_metric_value=best_metric_value),
            )

            # print("Experiment history: {}".format(json.dumps(history.history, indent=4)))
            # print("Experiment epoch: {}".format(json.dumps(history.epoch, indent=4)))
            print("Best value {}: {}".format(self.search_metric, best_metric_value))

            if best_experiment_num is None:
                best_experiment_num = experiment_num
            elif best_metric_value > experiments[best_experiment_num]['best_metric_value']:
                best_experiment_num = experiment_num

        print("Best experiment: {}".format(best_experiment_num))
        print("Experiment params: {}".format(experiments[best_experiment_num]))


def experiment(train_data_set, validation_data_set):

    gs = GridSearch(build_fn=create_srcnn, train_data_set=train_data_set, validation_data_set=validation_data_set,
                    grid_params={
                        'f1': [9, 18],
                        'f2': [1, 2],
                        'f3': [5, 10]
                    }, nb_epochs=10, search_metric='val_PSNRLoss')
    gs.run()
