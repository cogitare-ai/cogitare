from cogitare import utils
import torch


@utils.tensorfy(0, 1, tensor_klass=torch.LongTensor)
def accuracy(prediction, expected):
    """Computes the accuracy of the classification.

    Given the label prediction and the expected labels, returns the accuracy of
    the prediction.

    Args:
        prediction (torch.Tensor): tensor where each element represents the prediction
            for each sample.
        expected (torch.Tensor): with the same shape that prediction, contains the true labels
            for each sample.

    Shape:
        prediction: :math:`(N, S)`, or :math:`(S)`, where :math:`N` is the
            number of batches and :math:`S` is the number of samples per batch.
        prediction: must have the same shape that ``prediction``.
        output: :math:`(N)` when the input contains the batch dimension,
            :math:`(1)` otherwise.

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
    utils.assert_dim(prediction, "prediction", (1, 2))
    utils.assert_dim(expected, "expected", (1, 2))
    utils.assert_raise(prediction.size() == expected.size(), ValueError, '"prediction" and "expected" must have'
                       ' the same dimension.')
    if prediction.dim() == 1:
        prediction = prediction.view(1, -1)
        expected = expected.view(1, -1)

    eq = torch.eq(prediction, expected).float()
    correct = torch.sum(eq, 1)
    return correct / prediction.size(1)
