from cogitare import utils
from cogitare.core import PluginInterface
import matplotlib.pyplot as plt
import numpy as np


class PlottingMatplotlib(PluginInterface):
    """
    This plugin is a matplotlib interface to plot training and validation error
    during the training process.

    It's useful to monitor the model performance, overfitting, generalization, and so on.

    It's recommended to use this plugin at the **on_end_epoch** hook since the validation loss
    is calculated at this point.

    .. image:: _static/plugin_matplotlib.png
        :target: _static/plugin_matplotlib.png

    .. note:: If you want to plot the validation error, make sure to include the
        validation dataset in the model training.

    Args:
        source (str): determines the plot data. "train" to plot training error, "validation"
            for validation error, or "both" to include both losses in the plot.
        training_label (str): the label in the plot of the training error
        validation_label (str): the label in the plot of the validation error
        style (str): matplotlib style
        xlabel (str): label in the x-axis
        ylabel (str): label in the y-axis
        title (str): plot title
        show_std (bool): if True, display the standard deviation of the loss in the plot.

    Example::

        plot = PlottingMatplotlib(source='both')
        model.register_plugin(plot, 'on_end_epoch')
    """

    def __init__(self, source='train', training_label='Training loss', validation_label='Validation loss',
                 style='ggplot', xlabel='Epochs', ylabel='Loss', title='Training loss per epoch',
                 show_std=True):
        super(PlottingMatplotlib, self).__init__()
        utils.assert_raise(source in ('train', 'validation', 'both'), ValueError,
                           '"source" must be: "train", "validation" or "both"')
        self.source = source
        self.max_epochs = None
        self.style = style
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.training_label = training_label
        self.validation_label = validation_label
        self.title = title

        self.line = {
            'loss_mean': None,
            'loss_mean_validation': None
        }

        self.line_std = {
            'loss_mean': None,
            'loss_mean_validation': None
        }

        self.labels = {
            'loss_mean': self.training_label,
            'loss_mean_validation': self.validation_label
        }

        self.list_values = {
            'loss_mean': 'losses',
            'loss_mean_validation': 'losses_validation'
        }

        self.colors = {
            'loss_mean': 'blue',
            'loss_mean_validation': 'green'
        }

    def _create(self):
        plt.ion()
        plt.style.use(self.style)
        self.fig, self.ax = plt.subplots()

        plt.title(self.title)
        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)

        self.y = {
            'loss_mean': [0] * self.max_epochs,
            'loss_mean_validation': [0] * self.max_epochs
        }

        self.y_inf = {
            'loss_mean': [0] * self.max_epochs,
            'loss_mean_validation': [0] * self.max_epochs
        }

        self.y_sup = {
            'loss_mean': [0] * self.max_epochs,
            'loss_mean_validation': [0] * self.max_epochs
        }

        if self.source == 'train':
            del self.y['loss_mean_validation']
        elif self.source == 'validation':
            del self.y['loss_mean']

    def function(self, current_epoch, max_epochs, *args, **kwargs):
        if self.max_epochs is None:
            self.max_epochs = max_epochs
            self.x = list(range(1, max_epochs + 1))
            self._create()
        max_y = 0

        for d in self.y.keys():
            data = self.y[d]
            data_inf = self.y_inf[d]
            data_sup = self.y_sup[d]
            if d == 'loss_mean':
                std = np.std(kwargs['losses'])
            else:
                std = np.std(kwargs['losses_validation'])

            for pos in range(current_epoch - 1, self.max_epochs):
                data[pos] = kwargs[d]
                data_inf[pos] = kwargs[d] - std
                data_sup[pos] = kwargs[d] + std

            if self.line[d] is None:
                self.line[d], = self.ax.plot(self.x, self.y[d], '.-', label=self.labels[d],
                                             color=self.colors[d])
                self.ax.legend(loc='upper right')

            if self.line_std[d] is not None:
                self.line_std[d].remove()

            self.line_std[d] = self.ax.fill_between(self.x, self.y_inf[d], self.y_sup[d],
                                                    color=self.colors[d], alpha=0.1)

            max_y = max(max_y, max(self.y[d]))
            self.line[d].set_ydata(self.y[d])
        self.ax.set_ylim([0, max_y])
        self.fig.canvas.draw()
