from tqdm import tqdm


class ProgressBarFeedback(object):

    def __init__(self, total, *args, **kwargs):
        self.total = total
        self.bar = tqdm(total=total, *args, **kwargs)

    def __call__(self, idx=None, *args, **kwargs):
        if idx == self.total:
            self.bar.last_print_n = 1
            self.bar.n = 1
        self.bar.update()
