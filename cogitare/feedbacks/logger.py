import math
import time
from tqdm import tqdm
import sys


class LoggerFeedback(object):

    def __init__(self, title='[Logger]', msg='Loss: %.06f', show_time=True, output_file=None):
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

    def __call__(self, loss, *args, **kwargs):
        log = '%s %s %s' % (self.title, self.msg, self._time_spent())
        log = log % loss

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
