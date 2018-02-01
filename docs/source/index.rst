Welcome to Cogitare's documentation!
====================================


.. image:: _static/logo-line.png

**Cogitare** is a Modern, Fast, and Modular Deep Learning and Machine Learning framework in Python. A friendly interface for beginners and a powerful toolset for experts.

It uses the best of `PyTorch <http://pytorch.org/>`_, `Dask <https://dask.pydata.org/>`_, `NumPy <http://www.numpy.org/>`_, and others tools through a simple interface to train, to evaluate, to test
models and more with high performance.

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

Contributions are welcome!

.. toctree::
    :includehidden:
    :maxdepth: 2
    :caption: Introduction

    Home <self>
    installation
    quickstart
    tutorials

.. toctree::
    :includehidden:
    :maxdepth: 3
    :caption: Cogitare
    
    model
    sequential_model
    data
    sequential_data
    async_data
    plugins
    monitor
    metrics
    utils

.. toctree::
    :includehidden:
    :maxdepth: 3
    :caption: Models

    models/classic

.. toctree::
    :includehidden:
    :maxdepth: 3
    :caption: Extra

    contribute



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
