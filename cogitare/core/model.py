import torch
from torch import nn
from cogitare.utils import not_training, training, call_feedback, call_watchdog
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from torch.autograd import Variable
from cogitare.feedbacks import LoggerFeedback, ProgressBarFeedback, PlottingFeedback
from cogitare import utils


@add_metaclass(ABCMeta)
class Model(nn.Module):

    def __init__(self, cuda=None):
        super(Model, self).__init__()
        self._enables_cuda = utils.get_cuda(cuda)

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def loss(self):
        pass

    def _to_tensor(self, list):
        item = list[0]
        if torch.is_tensor(item):
            for l in list:
                l.unsqueeze_(0)
            return torch.cat(list)
        elif isinstance(item, int):
            return torch.LongTensor(list)
        elif isinstance(item, float):
            return torch.DoubleTensor(list)
        else:
            raise ValueError('Invalid data type: {}'.format(type(item)))

    @training
    def learn(self, dataset, optimizer, max_epochs=50, epoch_feedback='default',
              batch_feedback='default', epoch_watchdog=None, batch_watchdog=None):

        if self._enables_cuda:
            self.cuda()

        if epoch_feedback == 'default':
            epoch_feedback = [
                LoggerFeedback(title='[%s]' % self.__class__.__name__),
                ProgressBarFeedback(total=max_epochs, desc='epoch', leave=True),
                PlottingFeedback()
            ]

        if batch_feedback == 'default':
            batch_feedback = [
                ProgressBarFeedback(total=len(dataset), desc='batch', leave=True),
            ]

        for epoch in range(1, max_epochs + 1):
            total_loss = 0
            total_samples = 0
            batch_size = len(dataset)

            for idx, (input, target) in enumerate(dataset):
                idx += 1
                if isinstance(input, list):
                    input = self._to_tensor(input)

                if isinstance(target, list):
                    target = self._to_tensor(target)

                input = input.float()
                if self._enables_cuda:
                    input, target = input.cuda(), target.cuda()

                optimizer.zero_grad()

                input, target = Variable(input), Variable(target)

                output = self.forward(input)
                loss = self.loss(output, target)

                loss.backward()
                optimizer.step()

                loss = loss.data[0]
                total_loss += loss
                total_samples += 1
                call_feedback(batch_feedback, instance=self, idx=idx, input=input,
                              output=output, target=target, loss=loss, max_idx=batch_size)
                if call_watchdog(batch_watchdog) is True:
                    return False

            total_loss /= total_samples
            call_feedback(epoch_feedback, instance=self, idx=epoch, loss=total_loss,
                          max_idx=max_epochs)
            if call_watchdog(epoch_watchdog) is True:
                return False
        return True

    @not_training
    def predict(self, x):
        return self.forward(x)

    def evaluate(self):
        pass
