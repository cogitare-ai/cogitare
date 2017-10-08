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
        legend_pos (str): position of the label legends
        file_name (str): if set, save the plot image in the specified path.
        freq (int): the frequency to execute this model. The model will execute at each ``freq`` call.

    Example::


        plot = PlottingMatplotlib()
        # plot the training error with the std
        plot.add_variable('loss', 'Loss', color='blue', use_std=True)
        # plot the validation error without std
        plot.add_variable('on_end_batch_Evaluator_loss',
                          'Validation', color='green', use_std=False)

        model.register_plugin(plot, 'on_end_epoch')
    """
    def __init__(self, style='ggplot', xlabel='Epochs', ylabel='Loss', title='Training loss per epoch',
                 legend_pos='upper right', file_name=None, freq=1):
        super(PlottingMatplotlib, self).__init__(freq=freq)
        self._style = style
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._title = title
        self._legend_pos = legend_pos
        self._file_name = file_name

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

    def add_variable(self, name, label, style='.-', color=None, use_std=True):
        """Register a varible to be plotted.

        Args:
            name (str): the name of a state variable to plot. Read the :meth:`cogitare.Model.register_plugin`
                for available names. This variable must return a numeric value or a list of them.
            label (str): the label displayed in the plot
            style (str): line style (check matplotlib for more information).
            color (str): line color. If not defined, matplotlib will use a collor from its pallete.
            use_std (bool): if True, plots the standard deviation of the variable defined
                by ``name``. To compute the std, the ``name`` variable must be a list.

        Example::

            # plot the loss error with the std
            plot.add_variable('loss', 'Loss', color='blue')
        """

        plot = {
            'name': name,
            'label': label,
            'style': style,
            'use_std': use_std,
            'color': color,
            'data': [],
            'data_inf': [],
            'data_sup': [],

            'line': None,
            'line_std': None
        }

        self._variables.append(plot)

    def _plot_data(self, plot, y_mean, y, kwargs):
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

        if plot['use_std']:
            std = np.std(y)
            plot['data_inf'].append(y_mean - std)
            plot['data_sup'].append(y_mean + std)

            if plot['line_std']:
                plot['line_std'].remove()

            p = self._ax.fill_between(xdata, plot['data_inf'], plot['data_sup'],
                                      color=plot['line'].get_color(), alpha=0.1)
            plot['line_std'] = p

    def function(self, **kwargs):
        if not self._created:
            self._create_plot()
            self.max_y = -float('inf')
            self.min_y = float('inf')

        # xdata = None
        for plot in self._variables:
            y = kwargs[plot['name']]
            y_mean = float(np.mean(y))
            plot['data'].append(y_mean)

            self.max_y = max(self.max_y, y_mean)
            self.min_y = min(self.min_y, y_mean)
            self._plot_data(plot, y_mean, y, kwargs)

        self._ax.set_ylim([min(0, self.min_y), self.max_y])
        self._plt.xticks(np.arange(1, len(plot['data']) + 1))
        self._fig.canvas.draw()

        if self._file_name:
            self._fig.savefig(self._file_name)
