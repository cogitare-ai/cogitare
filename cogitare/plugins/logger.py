from cogitare.core import PluginInterface
import math
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
        title (str): title that appears in the beginning of the message.
        msg (str): string with the message formating. You can use this parameter
            to customize the logging variable and format. The message must be compatible with
            :meth:`str.format`, and it have access to the model state variable.
        show_time (bool): if True, appends the running time in the end of the message.
        output (file): if provided, write the log message in the file.
        freq (int): the frequency to execute this model.

    Examples::

        logger1 = Logger()
        logger2 = Logger(msg='Batch loss: {loss:.6f}')
        logger3 = Logger(msg='Validation loss: {validation_loss:.6f}')

        model.register_plugin([logger1, logger3], 'on_end_epoch')
        model.register_plugin(logger2, 'on_end_batch')
    """

    def __init__(self, title='[Logger]', msg='Loss: {loss:.6f}', show_time=True, output_file=None, freq=1):
        super(Logger, self).__init__(freq=freq)

        self.title = title
        self.msg = msg
        self.show_time = show_time
        self.output_file = output_file

        if show_time:
            self._start_time = time.time()

    def _time_spent(self):
        if not self.show_time:
            return ''

        seconds = time.time() - self._start_time
        minutes = math.floor(seconds / 60)
        seconds = seconds % 60
        return '%dm %ds' % (minutes, seconds)

    def function(self, *args, **kwargs):
        log = '%s %s %s' % (self.title, self.msg.format(**kwargs), self._time_spent())

        if hasattr(tqdm, '_instances') and tqdm._instances:
            for i in tqdm._instances:
                i.clear()
            sys.stderr.flush()
            sys.stdout.write(log + '\n')
            for i in tqdm._instances:
                i.refresh()
        else:
            print(log)

        if self.output_file:
            self.output_file.write(log + '\n')