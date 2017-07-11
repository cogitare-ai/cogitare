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

    def _to_tensor(self, input, tensor_klass=None):
        # if list, cast it to the compatible tensor type
        tensor = None

        if isinstance(input, list):
            item = input[0]

            if torch.is_tensor(item):
                for l in input:
                    l.unsqueeze_(0)
                tensor = torch.cat(input)
            elif isinstance(item, int):
                tensor = torch.LongTensor(input)
            elif isinstance(item, float):
                tensor = torch.DoubleTensor(input)
            else:
                raise ValueError('Invalid data type: {}'.format(type(item)))
        elif torch.is_tensor(input):
            tensor = input
        else:
            raise ValueError('Invalid data type: {}'.format(type(input)))

        if tensor_klass:
            tensor = tensor.type_as(tensor_klass())

        if self._enables_cuda:
            tensor = tensor.cuda()

        return tensor

    @training
    def learn(self, dataset, optimizer, max_epochs=50, epoch_feedback='default',
              batch_feedback='default', epoch_watchdog=None, batch_watchdog=None,
              input_type=None, target_type=None):

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
                input = Variable(self._to_tensor(input, input_type))
                target = Variable(self._to_tensor(target, target_type))

                optimizer.zero_grad()

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
