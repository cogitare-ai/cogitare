<p align="center"><img width="80%" src="https://raw.githubusercontent.com/cogitare-ai/cogitare/master/docs/source/art/logo-line.png" /></p>

# 1. Cogitare
**Cogitare** is a Modern, Fast, and Modular Deep Learning and Machine Learning framework in Python. A friendly interface for beginners and a powerful toolset for experts. 

http://docs.cogitare-ai.org/
	
[![Build Status](https://travis-ci.org/cogitare-ai/cogitare.svg?branch=master)](https://travis-ci.org/cogitare-ai/cogitare)
[![codecov](https://codecov.io/gh/cogitare-ai/cogitare/branch/master/graph/badge.svg)](https://codecov.io/gh/cogitare-ai/cogitare)
[![Code Climate](https://codeclimate.com/github/cogitare-ai/cogitare/badges/gpa.svg)](https://codeclimate.com/github/cogitare-ai/cogitare)


It uses the best of [PyTorch](http://pytorch.org/), [Dask](https://dask.pydata.org/), [NumPy](http://www.numpy.org/), and others tools through a simple interface to train, to evaluate, to test
models and more.

With Cogitare, you can use classical machine learning algorithms with high
performance and develop state-of-the-art models quickly.

The primary objectives of Cogitare are:

- provide an easy-to-use interface to train and evaluate models;
- provide tools to debug and analyze the model;
- provide implementations of state-of-the-art models (models for common tasks, ready
  to train and ready to use);
- provide ready-to-use implementations of straightforward and classical models (such as
  LogisticRegression);
- be compatible with models for a broad range of problems;
- be compatible with other tools (scikit-learn, etcs);
- keep growing with the community: accept as many new features as possible;
- provide a friendly interface to beginners, and powerful features for experts;
- take the best of the hardware through multi-processing and multi-threading;
- and others.

Currently, it's a work in progress project that aims to provide a complete
toolchain for machine learning and deep learning development, taking the best
of cuda and multi-core processing.

# 2. Install

- Install PyTorch from http://pytorch.org/
- Install Cogitare from PIP:

      pip install cogitare



# 3. Quickstart


This is a simple tutorial to get started with Cogitare main functionalities.

In this tutorial, we will write a Convolutional Neural Network (CNN) to
classify handwritten digits (MNIST).

### 3.1 Model

We start by defining our CNN model.

When developing a model with Cogitare, your model must extend the ``cogitare.Model`` class. This class provides the Model interface, which allows you to train and evaluate the model efficiently.

To implement a model, you must extend the ``cogitare.Model`` class and implement the ``forward()`` and ``loss()`` methods. The forward method will receive the batch. In this way, it is necessary to implement the forward pass through the network in this method, and then return the output of the net. The loss method will receive the output of the ``forward()`` and the batch received from the iterator, apply a loss function, compute and return it.

The Model interface will iterate over the dataset, and execute each batch on ``forward``, ``loss``, and ``backward``.


```python
# adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py
from cogitare import Model
from cogitare import utils
from cogitare.data import DataSet, AsyncDataLoader
from cogitare.plugins import EarlyStopping
from cogitare.metrics.classification import accuracy
import cogitare

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
import torch.optim as optim

from sklearn.datasets import fetch_mldata

import numpy as np

CUDA = True


cogitare.utils.set_cuda(CUDA)
```


```python
class CNN(Model):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        # define the model
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, batch):
        # in this sample, each batch will be a tuple containing (input_batch, expected_batch)
        # in forward in are only interested in input so that we can ignore the second item of the tuple
        input, _ = batch
        
        # batch X flat tensor -> batch X 1 channel (gray) X width X heigth
        input = input.view(32, 1, 28, 28)
        
        # pass the data in the net
        x = F.relu(F.max_pool2d(self.conv1(input), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        # return the model output
        return F.log_softmax(x, dim=1)
    
    def loss(self, output, batch):
        # in this sample, each batch will be a tuple containing (input_batch, expected_batch)
        # in loss in are only interested in expected so that we can ignore the first item of the tuple
        _, expected = batch
        
        return F.nll_loss(output, expected)
```

The model class is simple; it only requires de forward and loss methods. By default, Cogitare will backward the loss returned by the ``loss()`` method, and optimize the model parameters. If you want to disable the Cogitare backward and optimization steps, just return ``None`` in the loss function. If you return None, you are responsible by backwarding and optimizing the parameters.

### 3.2 Data Loading
In this step, we will load the data from sklearn package.


```python
mnist = fetch_mldata('MNIST original')
mnist.data = (mnist.data / 255).astype(np.float32)
```

Cogitare provides a toolbox to load and pre-process data for your models. In this introduction, we will use the ``DataSet`` and the ``AsyncDataLoader`` as examples.

The ``DataSet`` is responsible by iterating over multiples data iterators (in our case, we'll have two data iterators: input samples, expected samples).


```python
# as input, the DataSet is expected a list of iterators. In our case, the first iterator is the input 
# data and the second iterator is the target data

# also, we set the batch size to 32 and enable the shuffling

# drop the last batch if its size is different of 32
data = DataSet([mnist.data, mnist.target.astype(int)], batch_size=32, shuffle=True, drop_last=True)

# then, we split our dataset into a train and into a validation sets, by a ratio of 0.8
data_train, data_validation = data.split(0.8)
```

Notice that Cogitare accepts any iterator as input. Instead of using our DataSet, you can use the mnist.data itself, PyTorch's data loaders, or any other input that acts as an iterator.

In some cases, we can increase the model performance by loading the data using multiples threads/processes or by pre-loading the data before being requested by the model.

With the ``AsyncDataLoader``, we can load N batches ahead of the model execution in parallel. We present this technique in this sample because it can increase performance in a wide range of models (when the data loading or pre-processing is slower than the model execution).


```python
def pre_process(batch):
    input, expected = batch
    
    # the data is a numpy.ndarray (loaded from sklearn), so we need to convert it to Variable
    input = utils.to_variable(input, dtype=torch.FloatTensor)  # converts to a torch Variable of LongTensor
    expected = utils.to_variable(expected, dtype=torch.LongTensor)  # converts to a torch Variable of LongTensor
    return input, expected


# we wrap our data_train and data_validation iterators over the async data loader.
# each loader will load 16 batches ahead of the model execution using 8 workers (8 threads, in this case).
# for each batch, it will be pre-processed in parallel with the preprocess function, that will load the data
# on GPU
data_train = AsyncDataLoader(data_train, buffer_size=16, mode='threaded', workers=8, on_batch_loaded=pre_process)
data_validation = AsyncDataLoader(data_validation, buffer_size=16, mode='threaded', workers=8, on_batch_loaded=pre_process)
```

to cache the async buffer before training, we can:


```python
data_train.cache()
data_validation.cache()
```

## 3.3 Training

Now, we can train our model.

First, lets create the model instance and add the default plugins to watch the training status.
The default plugin includes:

- Progress bar per batch and epoch
- Plot training and validation losses (if validation_dataset is present)
- Log training loss


```python
model = CNN()
model.register_default_plugins()
```

Besides that, we may want to add some extra plugins, such as the EarlyStopping. So, if the model is not decreasing the loss after N epochs, the training stops and the best model is used.

To add the early stopping algorithm, you can use:


```python
early = EarlyStopping(max_tries=10, path='/tmp/model.pt')
# after 10 epochs without decreasing the loss, stop the training and the best model is saved at /tmp/model.pt

# the plugin will execute in the end of each epoch
model.register_plugin(early, 'on_end_epoch')
```

Also, a common technique is to clip the gradient during training. If you want to clip the grad, you can use:


```python
model.register_plugin(lambda *args, **kw: clip_grad_norm(model.parameters(), 1.0), 'before_step')
# will execute the clip_grad_norm before each optimization step
```

Now, we define the optimizator, and then start the model training:


```python
optimizer = optim.Adam(model.parameters(), lr=0.001)

if CUDA:
    model = model.cuda()
model.learn(data_train, optimizer, data_validation, max_epochs=100)
```

    2018-02-02 20:59:23 sprawl cogitare.core.model[2443] INFO Model: 
    
    CNN(
      (conv1): Conv2d (1, 10, kernel_size=(5, 5), stride=(1, 1))
      (conv2): Conv2d (10, 20, kernel_size=(5, 5), stride=(1, 1))
      (conv2_drop): Dropout2d(p=0.5)
      (fc1): Linear(in_features=320, out_features=50)
      (fc2): Linear(in_features=50, out_features=10)
    )
    
    2018-02-02 20:59:23 sprawl cogitare.core.model[2443] INFO Training data: 
    
    DataSet with:
        containers: [
            TensorHolder with 1750x32 samples
    	TensorHolder with 1750x32 samples
        ],
        batch size: 32
    
    
    2018-02-02 20:59:23 sprawl cogitare.core.model[2443] INFO Number of trainable parameters: 21,840
    2018-02-02 20:59:23 sprawl cogitare.core.model[2443] INFO Number of non-trainable parameters: 0
    2018-02-02 20:59:23 sprawl cogitare.core.model[2443] INFO Total number of parameters: 21,840
    2018-02-02 20:59:23 sprawl cogitare.core.model[2443] INFO Starting the training ...
    2018-02-02 21:02:04 sprawl cogitare.core.model[2443] INFO Training finished
    
    Stopping training after 10 tries. Best score 0.0909
    Model restored from: /tmp/model.pt

![](http://docs.cogitare-ai.org/_images/quickstart_23_3.png)


To check the model loss and accuracy on the validation dataset:


```python
def model_accuracy(output, data):
    _, indices = torch.max(output, 1)
    
    return accuracy(indices, data[1])

# evaluate the model loss and accuracy over the validation dataset
metrics = model.evaluate_with_metrics(data_validation, {'loss': model.metric_loss, 'accuracy': model_accuracy})

# the metrics is an dict mapping the metric name (loss or accuracy, in this sample) to a list of the accuracy output
# we have a measurement per batch. So, to have a value of the full dataset, we take the mean value:

metrics_mean = {'loss': 0, 'accuracy': 0}
for loss, acc in zip(metrics['loss'], metrics['accuracy']):
    metrics_mean['loss'] += loss
    metrics_mean['accuracy'] += acc.data[0]

qtd = len(metrics['loss'])

print('Loss: {}'.format(metrics_mean['loss'] / qtd))
print('Accuracy: {}'.format(metrics_mean['accuracy'] / qtd))
```

    Loss: 0.10143917564566948
    Accuracy: 0.9846252860411899


One of the advantages of Cogitare is the plug-and-play APIs, which let you add/remove functionalities easily. With this sample, we trained a model with training progress bar, error plotting, early stopping, grad clipping, and model evaluation easily.

Contribution
------------
Cogitare is a work in progress project, and any contribution is welcome.

You can contribute testing and providing bug reports, proposing feature ideas,
fixing bugs, pushing code, etcs.

1. You want to propose a new Feature and implement it
	- post about your intended feature, and we shall discuss the design and implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue
    - Look at the outstanding issues here: https://github.com/cogitare-ai/cogitare/issues
    - Pick an issue and comment on the task that you want to work on this feature
    - If you need more context on a particular issue, please ask and we shall provide.


Once you finish implementing a feature or bugfix, please send a Pull Request to
https://github.com/cogitare-ai/cogitare

If you are not familiar with creating a Pull Request, here are some guides:
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/
