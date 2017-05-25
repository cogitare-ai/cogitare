from torch import nn
from cogitare.utils import not_training, training
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from cogitare import utils
from torch.autograd import Variable


@add_metaclass(ABCMeta)
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def loss(self):
        pass

    @training
    def learn(self, data_loader, optimizer, max_epochs=50, epoch_feedback=None,
              batch_feedback=None):

        epoch_feedback = utils.get_epoch_feedback(epoch_feedback)
        batch_feedback = utils.get_batch_feedback(batch_feedback)

        for epoch in range(1, max_epochs + 1):
            total_loss = 0
            total_samples = 0

            for idx, (input, target) in enumerate(data_loader):
                if isinstance(input, list):
                    raise NotImplementedError

                optimizer.zero_grad()

                input, target = Variable(input), Variable(target)

                output = self.forward(input)
                loss = self.loss(output, target)

                loss.backward()
                optimizer.step()

                total_loss += loss
                total_samples += 1
                batch_feedback.update(instance=self, idx=idx, input=input,
                                      output=output, target=target, loss=loss)

            total_loss /= total_samples
            epoch_feedback.update(instance=self, idx=epoch, loss=total_loss)

    @not_training
    def predict(self, x):
        return self.forward(x)

    def evaluate(self):
        pass

    def add_callback(self):
        pass
