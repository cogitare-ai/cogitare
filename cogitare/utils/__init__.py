from cogitare import config
from cogitare.core.feedback import Feedback
from torch.nn import Module


def get_verbosity(verbose=None):
    """if verbose is None, returns the default verbosity from config
    """
    if verbose is None:
        return config['VERBOSE']
    else:
        return bool(verbose) is True


def not_training(func):
    """decorator do disable the training during execution. Must be used inside a Module class
    """
    def f(self, *args, **kwargs):
        assert isinstance(self, Module)
        default = self.training
        self.train(False)
        func(self, *args, **kwargs)
        self.train(default)

    return f


def training(func):
    """decorator do enable the training during execution. Must be used inside a Module class
    """
    def f(self, *args, **kwargs):
        assert isinstance(self, Module)
        default = self.training
        self.train(True)
        func(self, *args, **kwargs)
        self.train(default)

    return f


def get_epoch_feedback(feedback=None):
    """if feedback is None, returns the default epoch feedback system from the config.
    """
    if feedback is None:
        return config['FEEDBACK_EPOCH']
    else:
        assert isinstance(feedback, Feedback)
        return feedback


def get_batch_feedback(feedback=None):
    """if feedback is None, returns the default batch feedback system from the config.
    """
    if feedback is None:
        return config['FEEDBACK_BATCH']
    else:
        assert isinstance(feedback, Feedback)
        return feedback
