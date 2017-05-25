from cogitare.core.feedback import Feedback
import math
import time


class Logger(Feedback):

    def __init__(self, title='[Logger]', msg='Loss: %.06f', show_time=True):
        self.title = title
        self.msg = msg
        self.show_time = show_time

        if show_time:
            self._start_time = time.time()

    def time_spent(self):
        if not self.show_time:
            return ''

        seconds = time.time() - self._start_time
        minutes = math.floor(seconds / 60)
        seconds = seconds % 60
        return '%dm %ds' % (minutes, seconds)

    def update(self, loss, *args, **kwargs):
        log = '%s %s %s' % (self.title, self.msg, self.time_spent())
        print(log % loss)
