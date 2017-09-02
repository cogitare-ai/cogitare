from cogitare import utils
from cogitare.core import PluginInterface
import matplotlib.pyplot as plt


class PlottingMatplotlibFeedback(PluginInterface):

    def __init__(self, title, style='ggplot', *args, **kwargs):
        freq = kwargs.pop('freq', 1)
        super(PlottingMatplotlibFeedback, self).__init__(freq)

        plt.ion()
        plt.style.use(style)
        self.fig, self.ax = plt.subplots()
        plt.title(title)
        if 'ylabel' in kwargs:
            plt.ylabel(kwargs['ylabel'])
        if 'xlabel' in kwargs:
            plt.xlabel(kwargs['xlabel'])

        self.y = None
        self.line = None

    def function(self, current_epoch, loss, max_epochs, *args, **kwargs):
        if self.y is None:
            self.y = list(range(1, max_epochs + 1))

        for pos in range(current_epoch - 1, max_epochs):
            self.y[pos] = loss

        if self.line is None:
            self.line, = self.ax.plot(self.y, '.-')

        self.ax.set_ylim([0, max(self.y)])
        self.line.set_ydata(self.y)
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
