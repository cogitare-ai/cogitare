.. role:: hidden
    :class: hidden-section

Plugins
=======

Plugins are high-level objects (any callable object, such as classes with
``__call__``, user defined functions, and lambdas) that are used to inspect the
model during training, to provide feedbacks to the user, and to interact with the model.

For example, you can register a plugin to  plot train and validation loss
during each epoch, along with an early stopping algorithm that watches the
validation loss and saves/stops the model if the loss is not decreasing.

We start by exemplifying who to use your plugin and then describe the
plugins already implemented in Cogitare.

If you have a plugin implementation/idea/request, contributions are welcome!


Custom Plugin
-------------

To use your plugin with Cogitare, you just need to provide a callable
object when registering the plugin.

For example, if you want to apply a grad clipping, you can use:

Example::

    def norm(model, **kwargs):
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

    model.register_plugin(norm, 'before_step')


Register a plugin
-----------------

To register a plugin, use the :meth:`cogitare.Model.register_plugin` method.

Check the :meth:`~cogitare.Model.register_plugin` docs for more details about
the available events to watch, and which variables are available per event.

Official Plugins
----------------


:hidden:`EarlyStopping`
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: cogitare.plugins.EarlyStopping
    :members:

:hidden:`Evaluator`
~~~~~~~~~~~~~~~~~~~
.. autoclass:: cogitare.plugins.Evaluator
    :members:

:hidden:`Logger`
~~~~~~~~~~~~~~~~
.. autoclass:: cogitare.plugins.Logger
    :members:

:hidden:`PlottingMatplotlib`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: cogitare.plugins.PlottingMatplotlib
    :members:

:hidden:`ProgressBar`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: cogitare.plugins.ProgressBar
    :members:
