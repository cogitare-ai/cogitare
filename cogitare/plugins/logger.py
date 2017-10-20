from cogitare.core import PluginInterface
from dateutil.relativedelta import relativedelta
import logging
import coloredlogs
import time
from tqdm import tqdm
import sys


class Logger(PluginInterface):
    """
    The Logger plugin prints in the stdout (and in a file, optionally), a value from the
    model state.

    For example, it can be used to print the training error during each batch or epoch,
    or just print the number of samples processed so far.

    By default, it will print the loss error on the training data, but it can be configured using the
    ``msg`` parameter.

    .. image:: _static/plugins.png

    Args:
        title (str): title that appears at the beginning of the message.
        msg (str): string with the message formatting. You can use this parameter
            to customize the logging variable and format. The message must be compatible with
            :meth:`str.format`, and it has access to the model state variable.
        show_time (bool): if True, appends the running time at the end of the message.
        output (file): if provided, write the log message in the file.
        freq (int): the frequency to execute this model. The model will execute at each ``freq`` call.

    Examples::

        logger1 = Logger()
        logger2 = Logger(msg='Batch loss: {loss_mean:.6f}')
        logger3 = Logger(msg='Validation loss: {on_end_batch_Evalualor_loss:.6f}')

        model.register_plugin([logger1, logger3], 'on_end_epoch')
        model.register_plugin(logger2, 'on_end_batch')
    """

    def __init__(self, title='[Logger]', msg='Loss: {loss_mean:.6f}', show_time=True, output_file=None, freq=1):
        super(Logger, self).__init__(freq=freq)

        self.title = title
        self.msg = msg
        self.show_time = show_time
        self.output_file = output_file
        self.logger = logging.getLogger(title)
        coloredlogs.install(level='DEBUG', logger=self.logger)

        if show_time:
            self._start_time = time.time()

    def _time_spent(self):
        if not self.show_time:
            return ''

        intervals = ['days', 'hours', 'minutes', 'seconds']
        seconds = relativedelta(seconds=int(time.time() - self._start_time))
        time_str = ' '.join('{} {}'.format(getattr(seconds, k), k) for k in intervals
                            if getattr(seconds, k))

        return '| ' + time_str

    def function(self, *args, **kwargs):
        log = '%s %s %s' % (self.title, self.msg.format(**kwargs), self._time_spent())

        if getattr(tqdm, '_instances', None):
            for i in tqdm._instances:
                i.clear()
            sys.stderr.flush()
            self.logger.info(log)
            for i in tqdm._instances:
                i.refresh()
        else:
            self.logger.info(log)

        if self.output_file:
            self.output_file.write(log + '\n')
