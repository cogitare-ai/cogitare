from cogitare.core import PluginInterface
from cogitare.utils import StopTraining
import torch


class EarlyStopping(PluginInterface):

    def __init__(self, max_tries, path, **kwargs):
        super(EarlyStopping, self).__init__(**kwargs)
        self.max_tries = max_tries
        self.path = path

        self._best_epoch = 0
        self._best_score = float('inf')

    def function(self, model, validation_loss, current_epoch, *args, **kwargs):
        if validation_loss < self._best_score:
            self._best_score = validation_loss

            torch.save(model.state_dict(), self.path)
            self._best_epoch = current_epoch
        elif current_epoch - self._best_epoch > self.max_tries:
            model.load_state_dict(torch.load(self.path))
            print('\n\nStopping training after %d tries. Best score %.4f, restoring model from %s' % (
                self.max_tries, self._best_score, self.path))
            raise StopTraining
