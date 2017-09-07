from cogitare import utils
from cogitare.core import PluginInterface
import matplotlib.pyplot as plt


class PlottingMatplotlib(PluginInterface):
    """
    This plugin is a matplotlib interface to plot training and validation error
    during the training process.

    It's useful to monitor the model performance, overfitting, generalization, and so on.

    It's recommended to use this plugin at **on_end_epoch** hook, since the validation loss
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

    Example::

        plot = PlottingMatplotlib(source='both')
        model.register_plugin(plot, 'on_end_epoch')
    """

    def __init__(self, source='train', training_label='Training loss', validation_label='Validation loss',
                 style='ggplot', xlabel='Epochs', ylabel='Loss', title='Training loss per epoch'):
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
            'loss': None,
            'validation_loss': None
        }

        self.labels = {
            'loss': self.training_label,
            'validation_loss': self.validation_label
        }

    def _create(self):
        plt.ion()
        plt.style.use(self.style)
        self.fig, self.ax = plt.subplots()

        plt.title(self.title)
        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)

        self.y = {
            'loss': [0] * self.max_epochs,
            'validation_loss': [0] * self.max_epochs
        }

        if self.source == 'train':
            del self.y['validation_loss']
        elif self.source == 'validation':
            del self.y['train']

    def function(self, current_epoch, max_epochs, *args, **kwargs):
        if self.max_epochs is None:
            self.max_epochs = max_epochs
            self._create()
        max_y = 0

        for d in self.y.keys():
            data = self.y[d]

            for pos in range(current_epoch - 1, self.max_epochs):
                data[pos] = kwargs[d]

            if self.line[d] is None:
                self.line[d], = self.ax.plot(self.y[d], '.-', label=self.labels[d])
                self.ax.legend(loc='upper right')

            max_y = max(max_y, max(self.y[d]))
            self.line[d].set_ydata(self.y[d])
        self.ax.set_ylim([0, max_y])
        self.fig.canvas.draw()
