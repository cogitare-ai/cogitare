from cogitare.core import PluginInterface
import numpy as np
import matplotlib.pyplot as plt


class PlottingMatplotlib(PluginInterface):
    """
    This plugin is a matplotlib interface to plot the training state, such as the
    training and validation error during the training process.

    It's useful to monitor the model performance, overfitting, generalization, and so on.

    .. image:: _static/plugin_matplotlib.png
        :target: _static/plugin_matplotlib.png

    Args:
        style (str): matplotlib style
        xlabel (str): label in the x-axis
        ylabel (str): label in the y-axis
        title (str): plot title

    Example::

        plot = PlottingMatplotlib(source='both')
        model.register_plugin(plot, 'on_end_epoch')
    """
    def __init__(self, style='ggplot', xlabel='Epochs', ylabel='Loss', title='Training loss per epoch',
                 legend_pos='upper right'):
        super(PlottingMatplotlib, self).__init__()
        self._style = style
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._title = title
        self._legend_pos = legend_pos

        self._plt = None
        self._fig, self._ax = None, None
        self._variables = []
        self._created = False

    def _create_plot(self):
        plt.ion()
        plt.style.use(self._style)
        fig, ax = plt.subplots()

        plt.title(self._title)
        plt.xlabel(self._xlabel)
        plt.ylabel(self._ylabel)

        self._plt = plt
        self._fig = fig
        self._ax = ax

        self._created = True

    def add_variable(self, name, label, style='.-', color=None, std_data=None):
        plot = {
            'name': name,
            'label': label,
            'style': style,
            'std_data': std_data,
            'color': color,
            'data': [],
            'data_inf': [],
            'data_sup': [],

            'line': None,
            'line_std': None
        }

        self._variables.append(plot)

    def _plot_data(self, plot, y, kwargs):
        qtd = len(plot['data'])
        xdata = list(range(1, qtd + 1))

        if qtd == 1:  # first one
            plot['line'], = self._ax.plot(xdata, plot['data'], plot['style'],
                                          label=plot['label'],
                                          color=plot['color'])
            self._ax.legend(loc=self._legend_pos)
        else:
            plot['line'].set_ydata(plot['data'])
            plot['line'].set_xdata(xdata)

        if plot['std_data'] is not None:
            std_data = kwargs.get(plot['std_data'])
            std = np.std(std_data)
            plot['data_inf'].append(y - std)
            plot['data_sup'].append(y + std)

            if plot['line_std']:
                plot['line_std'].remove()

            p = self._ax.fill_between(xdata, plot['data_inf'], plot['data_sup'],
                                      color=plot['line'].get_color(), alpha=0.1)
            plot['line_std'] = p

    def function(self, **kwargs):
        if not self._created:
            self._create_plot()
        max_y = -float('inf')
        min_y = float('inf')

        # xdata = None
        for plot in self._variables:
            y = kwargs.get(plot['name'])
            plot['data'].append(y)

            max_y = max(max_y, max(plot['data']))
            min_y = min(min_y, min(plot['data']))
            self._plot_data(plot, y, kwargs)

        self._ax.set_ylim([min(0, min_y), max_y])
        self._plt.xticks(np.arange(1, len(plot['data']) + 1))
        self._fig.canvas.draw()
