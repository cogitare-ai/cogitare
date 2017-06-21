from cogitare import utils
import matplotlib.pyplot as plt


class PlottingMatplotlibFeedback(object):

    def __init__(self, title, style='ggplot', *args, **kwargs):
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

    def __call__(self, idx, loss, max_idx, *args, **kwargs):
        if self.y is None:
            self.y = list(range(1, max_idx + 1))

        for pos in range(idx - 1, max_idx):
            self.y[pos] = loss

        if self.line is None:
            self.line, = self.ax.plot(self.y, '.-')

        self.ax.set_ylim([0, max(self.y)])
        self.line.set_ydata(self.y)
        # plt.show()
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
