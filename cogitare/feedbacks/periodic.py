from cogitare.core.feedback import Feedback


class PeriodicFeedback(Feedback):

    def __init__(self, interval, feedback):
        self.interval = interval
        self._current_iter = 0
        assert isinstance(feedback, Feedback)

    def update(self, *args, **kwargs):
        self._current_iter += 1
        if self._current_iter % self.interval == 0:
            self.feedback.update(*args, **kwargs)
