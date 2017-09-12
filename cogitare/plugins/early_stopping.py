from cogitare.core import PluginInterface
from cogitare.utils import StopTraining


class EarlyStopping(PluginInterface):
    """
    This plugin provides the early stopping algorithm.

    During training, it will check the validation loss and if after ``max_tries``
    epochs the validation loss does not decrease, the training stops.

    The parameters with the best score found is saved in disk using :func:`torch.save` in the
    provided ``path``.

    When stopping, the plugin will automatically restore the model to the best checkpoint. If
    you don't want to have this feature, set ``restore_checkpoint=False``.

    It's recommended to use this plugin at the **on_end_epoch** hook since the validation loss
    is calculated at this point.


    Args:
        max_tries (int): number of epochs before stopping the training
        path (str): path to save the best model
        restore_checkpoint (bool): if True,  restore the model parameters after stopping by
            early-stopping.
        freq (int): the frequency to execute this model.

    Example::

        early = EarlyStopping(max_tries=10, path='/tmp/model.pt')
        model.register_plugin(early, 'on_end_epoch')
    """

    def __init__(self, max_tries, path, restore_checkpoint=True, freq=1):
        super(EarlyStopping, self).__init__(freq=freq)
        self.max_tries = max_tries
        self.path = path
        self.restore_checkpoint = restore_checkpoint

        self._best_epoch = 0
        self._best_score = float('inf')

    def function(self, model, loss_mean_validation, current_epoch, *args, **kwargs):
        if loss_mean_validation < self._best_score:
            self._best_score = loss_mean_validation

            model.save(self.path)
            self._best_epoch = current_epoch
        elif current_epoch - self._best_epoch > self.max_tries:
            print('\n\nStopping training after %d tries. Best score %.4f' % (
                self.max_tries, self._best_score))

            if self.restore_checkpoint:
                model.load(self.path)
                print('Model restored from: ' + self.path)

            raise StopTraining
