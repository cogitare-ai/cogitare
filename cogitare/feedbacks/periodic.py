from cogitare import utils


class PeriodicFeedback(object):

    def __init__(self, interval, feedback):
        utils.assert_raise(interval < 1, ValueError, 'Interval must greater or equal to 1')
        self.interval = interval
        self.feedback = feedback
        self._current_iter = 0

    def __call__(self, *args, **kwargs):
        self._current_iter += 1
        if self._current_iter % self.interval == 0:
            utils.call_feedback(self.feedback)
