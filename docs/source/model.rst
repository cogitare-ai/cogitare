cogitare.Model
==============


Implementing a Model
--------------------

To implement a model, you must extend the :class:`cogitare.Model` class and
implement the :meth:`cogitare.Model.forward` and :meth:`cogitare.Model.loss`
methods.


The forward loss will receive the batch. In this way, it is necessary to implement the forward
pass through the network in this method, and then return the output of the net.

The loss function will receive the output of the :meth:`cogitare.Model.forward`
and the batch received from iterator, apply a loss function, compute and return it.


As a simple example, to implement a `LogisticRegression` model with dropout in
the input layer, it can be implemented as follows::

	class LogisticRegression(Model):

		def __init__(self, input_size, num_classes=2, dropout=0):
			super(LogisticRegression, self).__init__()

			utils.assert_raise(num_classes >= 2, ValueError,
				'"num_classes" must be greater than or equal 2')
			utils.assert_raise(0 <= dropout < 1, ValueError,
				'"dropout" value must be between 0 and 1')
			utils.assert_raise(input_size >= 1, ValueError,
				'"input_size" value must be greater than or equal 1')
			self.arguments = {
				'input_size': input_size,
				'num_classes': num_classes,
				'dropout': dropout,
			}

			self.linear = nn.Linear(input_size, num_classes)

		def forward(self, sample):
			x = Variable(utils.to_tensor(sample[0], torch.FloatTensor, self.use_cuda))
			x = x.view(x.size(0), -1)
			data = F.dropout(x, self.arguments['dropout'])
			out = self.linear(data)
			return F.log_softmax(out)

		def loss(self, output, sample):
			expected = Variable(utils.to_tensor(sample[1], torch.LongTensor, self.use_cuda))

			return F.nll_loss(output, expected)

Notice that in the implementation above, it expects that each batch is composed of a tuple
`(x_data, y_data)`.

And then this model can be trained using::

	import torch.optim as optim
	import cogitare
	from cogitare.models.linear.logistic import LogisticRegression
	from cogitare.data import DataSet
	from sklearn.datasets import fetch_mldata

	# Data
	mnist = fetch_mldata('MNIST original')
	mnist.data = mnist.data / 255
	data = DataSet([mnist.data, mnist.target.astype(int)], batch_size=args.batch_size)
	data_train, data_validation = data.split(0.8)

	# Model
	l = LogisticRegression(input_size=784, num_classes=10, use_cuda=True)
	l.register_default_plugins()
	optimizer = optim.SGD(l.parameters(), lr=0.1, momentum=0.9)

	l.learn(data_train, optimizer, data_validation)
	l.save('model.pt')

	print('Model trainined!')
	input()

To use this model latter, you can load it from disk using::

    l = LogisticRegression(input_size=784, num_classes=10, use_cuda=True)
    l.load('model.pt')

and make new predictions using::

    output = l.predict(...)


As you can see, the model development is pretty similary to develop a pure
PyTorch model. To use a PyTorch model with Cogitare, you just need to extend
the :class:`cogitare.Model` instead of :class:`torch.nn.Module`, and put the
loss function inside the :meth:`cogitare.Model.loss` method.

With these simple modifications, you can use Cogitare to train and evaluate
your model quickly.

API
---

.. autoclass:: cogitare.Model
    :members:
