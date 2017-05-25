from cogitare.core.feedback import Feedback


class PipelineFeedback(Feedback):

    def __init__(self, feedbacks=None):
        if feedbacks is None:
            feedbacks = []
        else:
            assert isinstance(feedbacks, list)
            for f in feedbacks:
                assert isinstance(f, Feedback)
        self.feedbacks = feedbacks

    def update(self, *args, **kwargs):
        [f.update(*args, **kwargs) for f in self.feedbacks]
