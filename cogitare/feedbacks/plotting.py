from cogitare import utils
from cogitare.core import PluginInterface
import matplotlib.pyplot as plt


class PlottingMatplotlibFeedback(PluginInterface):

    def __init__(self, data_source, max_epochs, title, data_label=None, style='ggplot', *args, **kwargs):
        freq = kwargs.pop('freq', 1)
        super(PlottingMatplotlibFeedback, self).__init__(freq)

        self.data_source = data_source
        self.max_epochs = max_epochs

        if data_label is None:
            data_label = dict((d, d) for d in data_source)

        self.data_label = data_label

        plt.ion()
        plt.style.use(style)
        self.fig, self.ax = plt.subplots()
        plt.title(title)
        if 'ylabel' in kwargs:
            plt.ylabel(kwargs['ylabel'])
        if 'xlabel' in kwargs:
            plt.xlabel(kwargs['xlabel'])

        self.y = dict((d, [0] * max_epochs) for d in data_source)
        self.line = dict((d, None) for d in data_source)

    def function(self, current_epoch, *args, **kwargs):
        max_y = 0
        for d in self.data_source:
            if d not in kwargs:
                continue

            if kwargs[d] is None:
                continue

            data = self.y[d]

            for pos in range(current_epoch - 1, self.max_epochs):
                data[pos] = kwargs[d]

            if self.line[d] is None:
                self.line[d], = self.ax.plot(self.y[d], '.-', label=self.data_label[d])
                self.ax.legend(loc='upper right')

            max_y = max(max_y, max(self.y[d]))
            self.line[d].set_ydata(self.y[d])
        self.ax.set_ylim([0, max_y])
        self.fig.canvas.draw()


def PlottingFeedback(backend='matplotlib', *args, **kwargs):
    _backends = {
        'matplotlib': PlottingMatplotlibFeedback
    }
    utils.assert_raise(backend in _backends.keys(), ValueError,
                       '"backend" must be one of: ' + ','.join(_backends.keys()))

    kwargs.setdefault('title', 'Training loss per epoch')
    kwargs.setdefault('xlabel', 'Epochs')
    kwargs.setdefault('ylabel', 'Loss')

    return _backends[backend](*args, **kwargs)
