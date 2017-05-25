from cogitare.core.feedback import Feedback
from tqdm import tqdm


class ProgressBarFeedback(Feedback):

    def __init__(self, total, description, position):
        self.bar = tqdm(total=total, description=description, position=position)

    def update(self, *args, **kwargs):
        self.bar.update()
