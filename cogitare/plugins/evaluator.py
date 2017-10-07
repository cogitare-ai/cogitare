from cogitare.core import PluginInterface


class Evaluator(PluginInterface):
    """The Evaluator plugin is a shortcut to evaluate the model performance
    during training throught different metrics using the
    :meth:`cogitare.Model.evaluate_with_metrics` method.

    The results of the metrics are stored in the model state, and can be displayed
    using plots, logs, and etc.

    Args:
        dataset: the dataset to evaluate the model. Must have the same caracteristics that
            the training dataset, used in :meth:`cogitare.Model.learn`.
        metrics (dict): a dict mapping from {metric_name: metric_function}. A metric function
            receives two parameters: (the model output, the dataset batch)
        freq (int): the frequency to execute this model. The model will execute at each ``freq`` call.

    Returns:
        output (dict): a dict mapping metric_name -> to a list with the metric result for each batch.
        This is the :meth:`cogitare.Model.evaluate_with_metrics` output.

    Example::

        metrics = {
            'loss': model.metric_loss,
            'precision': precision
        }

        plugin = Evaluator(validation_dataset, metrics)
        model.register_plugin(plugin, 'on_end_batch')

        # ... if you want to plot these results
        plot = PlottingMatplotlib()
        plot.add_variable('on_end_batch_Evaluator_loss', 'Loss')
        plot.add_variable('on_end_batch_Evaluator_precision', 'Precision')
    """

    def __init__(self, dataset, metrics, freq=1):
        super(Evaluator, self).__init__(freq=freq)
        self.metrics = metrics
        self.dataset = dataset

    def function(self, model, *args, **kwargs):
        return model.evaluate_with_metrics(self.dataset, self.metrics)
