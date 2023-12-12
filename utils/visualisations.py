import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

class MetricsOverEpochsViz():

    def __init__(self, train_val_metrics, other_metrics, **kwargs):
        '''
        :param train_val_metrics: list; list of metrics which will be tracked
            both train and val sets (plotted on same axis)
        :param other_metrics: list; list of additional metrics which will
            be tracked individually (plotted on individual axis)
        '''

        self.epochs = 0
        train_metrics = {m:[] for m in train_val_metrics}
        val_metrics = {m:[] for m in train_val_metrics}
        other_metrics = {m:[] for m in other_metrics}
        self.metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'other': other_metrics
        }

    def add_metric(self, metric, type):

        '''
        :param metric: str; name of metric to add
        :param type: ['train', 'val', 'other']: whether it's a train, val or
            other metric
        '''

        self.metrics[type][metric] = []


    def add_value(self, metric, value, type):

        '''
        :param metric: str; name of metric
        :param value: float/int; value of metric
        :param type: ['train', 'val', 'other']: whether it's a train, val or
            other value
        '''
        assert type in self.metrics, "Invalid type of metric. Must be one of ['train', 'val', 'other']"
        assert metric in self.metrics[type], f"{metric} is not being tracked for {type}"

        self.metrics[type][metric].append(value)

    def step(self):

        '''
        Call every epoch to validate that each metric was registered for the
        epoch exactly once
        '''
        self.epochs += 1
        invalid_metrics = []
        for type in self.metrics:
            for metric in self.metrics[type]:
                if len(self.metrics[type][metric]) != self.epochs:
                    invalid_metrics.append((type, metric,
                                            len(self.metrics[type][metric])))

        assert not invalid_metrics, f"{self.epochs} epochs complete, but the following metrics had an invalid number of values: {invalid_metrics}"


    def save_values(self, outdir):

        with open(os.path.join(outdir, 'metrics_over_epochs_values.pkl'), 'wb') as fp:
            pickle.dump(self.metrics, fp)

    def plot_values(self, outdir):

        for metric in self.metrics['train']:
            train_vals = self.metrics['train'][metric]
            val_vals = self.metrics['val'][metric]

            plt.plot(train_vals)
            plt.plot(val_vals)
            plt.title(f"{metric} curve")
            plt.ylabel(metric)
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"train_val_{metric}.png"))
            plt.clf()

        for metric in self.metrics['other']:

            vals = self.metrics['other'][metric]
            plt.plot(vals)
            plt.title(f"{metric} curve")
            plt.ylabel(metric)
            plt.xlabel('epoch')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{metric}.png"))
            plt.clf()
