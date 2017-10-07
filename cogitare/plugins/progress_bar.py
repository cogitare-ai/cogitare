from tqdm import tqdm
from cogitare.core import PluginInterface
from cogitare import utils


class ProgressBar(PluginInterface):
    """This plugin adds a `tqdm <https://github.com/tqdm/tqdm>`_ progress bar to monitor the progress
    of the batch/epoch iterations.

    By default, the progress bar is set up to monitor epoch progress,
    but you can switch to batch iteration using the ``monitor`` variable.

    It's recommended to use this plugin at the **on_end_epoch** and the **on_end_bactch** hooks
    since the batch/epoch count is updated at this point.

    .. image:: _static/plugins.png

    Args:
        monitor (str): the state to be monitored. Must be ``epoch`` or ``batch``.
        desc (str): the name of the progress bar. If not provided, the monitor name
            will be used.
        freq (int): the frequency to execute this model. The model will execute at each ``freq`` call.
    Examples::

        bar1 = ProgressBar()
        bar2 = ProgressBar(monitor='batch')

        model.register_plugin(bar1, 'on_end_epoch')
        model.register_plugin(bar2, 'on_end_batch')
    """
    def __init__(self, monitor='epoch', desc=None, freq=1):
        super(ProgressBar, self).__init__(freq)
        if desc is None:
            desc = monitor
        utils.assert_raise(monitor in ('epoch', 'batch'), ValueError,
                           'Monitor must be one of: "epoch", "batch"')

        self._monitor = monitor
        self._desc = desc
        self._total = None
        self._bar = None
        self._var = None

    def function(self, *args, **kwargs):
        if self._bar is None:
            if self._monitor == 'epoch':
                self._total = kwargs['max_epochs']
                self._var = 'current_epoch'
            else:
                self._total = kwargs['num_batches']
                self._var = 'current_batch'

            self._bar = tqdm(total=self._total, desc=self._desc, leave=True)

        if kwargs[self._var] == self._total:
            self._bar.last_print_n = 1
            self._bar.n = 1

        self._bar.update()
