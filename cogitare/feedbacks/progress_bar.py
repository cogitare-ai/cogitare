from tqdm import tqdm
from cogitare.core import PluginInterface


class ProgressBarFeedback(PluginInterface):

    def __init__(self, total, indice_name, *args, **kwargs):
        freq = kwargs.pop('freq', 1)
        super(ProgressBarFeedback, self).__init__(freq)
        self.total = total
        self.indice_name = indice_name
        self.bar = tqdm(total=total, *args, **kwargs)

    def function(self, *args, **kwargs):
        if kwargs[self.indice_name] == self.total:
            self.bar.last_print_n = 1
            self.bar.n = 1
        self.bar.update()
