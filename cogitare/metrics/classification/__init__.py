from cogitare import utils
import torch


@utils.tensorfy(0, 1, dtype=torch.LongTensor)
def filter_labels(y, labels):
    """Utility used to create a mask to filter values in a tensor.

    Args:
        y (list, numpy.ndarray, torch.Tensor): tensor where each element is a numeric integer
            representing a label.
        labels (list, numpy.ndarray, torch.Tensor): filter used to generate the mask. For each
            value in ``y`` its mask will be "1" if its value is in ``labels``,
            "0" otherwise.

    Shape:
        y: can have any shape. Usually will be :math:`(N, S)` or :math:`(S)`,
            containing `batch X samples` or just a list of `samples`.
        labels: a flatten list, or a 1D LongTensor.

    Returns:
        mask (torch.ByteTensor): a binary mask, with "1" with the respective value from ``y`` is
        in the ``labels`` filter.

    Example::

        >>> a = torch.LongTensor([[1,2,3],[1,1,2],[3,5,1]])
        >>> a
         1  2  3
         1  1  2
         3  5  1
        [torch.LongTensor of size 3x3]
        >>> classification.filter_labels(a, [1, 2, 5])
         1  1  0
         1  1  1
         0  1  1
        [torch.ByteTensor of size 3x3]
        >>> classification.filter_labels(a, torch.LongTensor([1]))
         1  0  0
         1  1  0
         0  0  1
        [torch.ByteTensor of size 3x3]
    """
    mapping = torch.zeros(y.size()).byte()

    for label in labels:
        mapping = mapping | y.eq(label)

    return mapping


@utils.tensorfy(0, 1, dtype=torch.LongTensor)
def accuracy(prediction, expected, labels=None):
    """Computes the accuracy of the classification.

    Given the label prediction and the expected labels, returns the accuracy of
    the prediction.

    Args:
        prediction (list, numpy.ndarray, torch.Tensor): tensor where each element represents the prediction
            for each sample.
        expected (list, numpy.ndarray, torch.Tensor): with the same shape that prediction, contains the true labels
            for each sample.
        labels (list, numpy.ndarray, torch.Tensor): if defined, consider only the labels
            in this list to compute the metric. This is useful for ignoring labels with high
            frequency in the model.

    Shape:
        prediction: :math:`(N, S)`, or :math:`(S)`, where :math:`N` is the
            number of batches and :math:`S` is the number of samples per batch.
        prediction: must have the same shape that ``prediction``.
        output: :math:`(N)` when the input contains the batch dimension,
            a float otherwise.
        labels: flatten list, or 1D tensor.

    Returns:
        output (torch.FloatTensor): tensor with the accuracy for each batch.

    Example:

        >>> from cogitare.metrics import classification as C
        >>> C.accuracy([1, 2, 3], [3, 5, 3])
         0.3333
        [torch.FloatTensor of size 1x1]

    Example:

        >>> a = torch.bernoulli(torch.Tensor(3, 5).uniform_(0, 1))
        >>> b = torch.bernoulli(torch.Tensor(3, 5).uniform_(0, 1))
        >>> a
         1  0  1  1  0
         1  1  0  0  1
         1  1  1  0  0
        [torch.FloatTensor of size 3x5]
        >>> b
         1  1  0  0  0
         1  0  0  1  0
         1  0  1  0  1
        [torch.FloatTensor of size 3x5]
        >>> C.accuracy(a, b)
         0.4000
         0.4000
         0.6000
        [torch.FloatTensor of size 3x1]
    """
    params = ((prediction, 'prediction'), (expected, 'expected'))

    prediction, expected = utils._as_2d(*params)
    utils._assert_same_dim(*params)

    eq = torch.eq(prediction, expected).byte()
    if labels is None:
        correct = torch.sum(eq, 1).float()
        result = correct / prediction.size(1)
    else:
        mask = filter_labels(expected, labels)
        eq &= mask
        correct = torch.sum(eq, 1).float()
        result = correct / (mask.sum(1).float())

    return result.squeeze()
