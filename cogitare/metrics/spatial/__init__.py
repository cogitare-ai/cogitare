from cogitare import utils
import torch


@utils.tensorfy(0, 1)
def cosine_similarity(u, v, dim=1, eps=1e-8):
    """Computes the Cosine similarity between vectors along a dimension.

    Consine similarity is defined as:

        :math:`cosine\_similarity(u, v) = \dfrac{u \cdot v}{max(||u||_2 * ||v||_2, eps)}`

    Args:
        u (torch.Tensor): tensor 1
        v (torch.Tensor): tensor 2
        dim (int): the dimension to compare the vectors.
        eps (:obj:`float`): small value to avoid division by zero.

    Shape:
        u (\*, D, \*): where D is the provided dimension ``dim``.
        v (\*, D, \*): where D is the provided dimension ``dim``. ``u`` and ``v`` must have the same shape.
        output (\*, \*): same as input, withoud the ``dim`` dimension, and squeezed.

    Example::

        >>> x1 = torch.Tensor([1, 2, 3])
        >>> x2 = torch.Tensor([2, 2, 3])
        >>> cosine_similarity(x1, x2, dim=0)
         0.9723
        [torch.FloatTensor of size 1]

    Example::

        >>> a = torch.rand(3, 4, 5)
        >>> b = torch.rand(3, 4, 5)
        >>> cosine_similarity(a, b, dim=0)
         0.9588  0.8099  0.9444  0.6813  0.8883
         0.4378  0.9690  0.9900  0.8366  0.8544
         0.5790  0.8270  0.9120  0.9115  0.5138
         0.4563  0.9510  0.9207  0.6498  0.7282
        [torch.FloatTensor of size 4x5]
        >>> cosine_similarity(a, b, dim=2)
         0.9491  0.7288  0.7040  0.6721
         0.6425  0.8440  0.5450  0.5586
         0.9114  0.7884  0.6583  0.7154
        [torch.FloatTensor of size 3x4]

    References:
        - https://nlp.stanford.edu/IR-book/html/htmledition/dot-products-1.html
        - https://en.wikipedia.org/wiki/Cosine_similarity

    """
    params = ((u, 'u'), (v, 'v'))
    utils._assert_same_dim(*params)

    w12 = torch.sum(u * v, dim)
    w1 = torch.norm(u, 2, dim)
    w2 = torch.norm(v, 2, dim)

    div = (w1 * w2).clamp(min=eps)

    return (w12 / div).squeeze()


def cosine_distance(*args, **kwargs):
    """Cosine distance is a shortcut for:

        :math:`cosine\_distance(u, v) = 1.0 - cosine\_similarity(u, v)`

    Check the :func:`~cogitare.metrics.spatial.cosine\_similarity` for more information.
    """
    return 1.0 - cosine_similarity(*args, **kwargs)


@utils.tensorfy(0, 1)
def norm_distance(u, v, dim=1, norm=1, eps=1e-8):
    """Computes the norm distance between vectors along a dimension.

    The norm distance between two vectors is defined as:

        :math:`norm\_dist(u, v, p) = \sqrt[p]{max(\sum_i(|u_i - v_i|^p), eps)}`

    where :math:`p` is defined by the ``norm`` parameter.

    Args:
        u (torch.Tensor): tensor 1
        v (torch.Tensor): tensor 2
        dim (int): the dimension to compare the vectors.
        norm (:obj:`float`): p-norm
        eps (:obj:`float`): small value to avoid division by zero.

    Shape:
        u (\*, D, \*): where D is the provided dimension ``dim``.
        v (\*, D, \*): where D is the provided dimension ``dim``. ``u`` and ``v`` must have the same shape.
        output (\*, \*): same as input, withoud the ``dim`` dimension, and squeezed.

    Example::

        >>> x1 = torch.Tensor([1, 2, 3])
        >>> x2 = torch.Tensor([2, 2, 5])
        >>> norm_distance(x1, x2, dim=0, norm=1)
         3
        [torch.FloatTensor of size 1]
        >>> norm_distance(x1, x2, dim=0, norm=3)
         2.0801
        [torch.FloatTensor of size 1]

    Example::

        >>> a = torch.rand(3, 4, 5)
        >>> b = torch.rand(3, 4, 5)
        >>> norm_distance(a, b, dim=1, norm=2)
         1.2678  0.8048  1.1087  0.9852  0.5751
         0.9778  1.0164  1.1546  0.9549  1.2127
         0.8448  0.9415  0.5060  0.9469  0.9281
        [torch.FloatTensor of size 3x5]

    References:
        - https://en.wikipedia.org/wiki/Distance
    """
    params = ((u, 'u'), (v, 'v'))
    utils._assert_same_dim(*params)

    diff = (u - v).abs_()
    diff.pow_(norm)

    cum = torch.sum(diff, dim).clamp_(min=eps)
    cum.pow_(1.0 / norm)

    return cum.squeeze()


def euclidian_distance(u, v, dim=1, eps=1e-8):
    """The Euclidian distance is a shortcut for :func:`~cogitare.metrics.spatial.norm_distance`
    with the norm=2.

    The Euclidian distance is defined as:

        :math:`euclidian\_distance(u, v) = \sqrt{max(\sum_i(|u_i - v_i|^2), eps)}`

    Check the :func:`~cogitare.metrics.spatial.norm\_distance` for more information.
    """

    return norm_distance(u, v, dim=dim, norm=2, eps=eps)


@utils.tensorfy(0, 1)
def manhattan_distance(u, v, dim=1):
    """The Manhattan Distance (aka :math:`L_1` distance, :math:`\ell_1` norm, taxicab distance, city block distance,
    and snake distance) of two vectors can be defined as:

        :math:`manhattan\_distance(u, v) = \sum_i|u_i - v_i|`

    This is an optimized version (and equivalent) of using :func:`norm_distance` with ``norm=1``.

    Args:
        u (torch.Tensor): tensor 1
        v (torch.Tensor): tensor 2
        dim (int): the dimension to compare the vectors.

    Shape:
        u (\*, D, \*): where D is the provided dimension ``dim``.
        v (\*, D, \*): where D is the provided dimension ``dim``. ``u`` and ``v`` must have the same shape.
        output (\*, \*): same as input, withoud the ``dim`` dimension, and squeezed.

    Example::

        >>> x1 = torch.Tensor([1, 2, 3])
        >>> x2 = torch.Tensor([2, 2, 5])
        >>> manhattan_distance(x1, x2, dim=0)
         3
        [torch.FloatTensor of size 1]

    Example::

        >>> a = torch.rand(3, 4, 5)
        >>> b = torch.rand(3, 4, 5)
        >>> manhattan_distance(a, b, dim=2)
         2.5895  1.4149  0.9355  1.8885
         2.7076  1.8408  1.8399  1.3157
         1.2883  2.5685  1.7510  2.0703
        [torch.FloatTensor of size 3x4]

    References:
        - https://en.wikipedia.org/wiki/Taxicab_geometry
    """
    params = ((u, 'u'), (v, 'v'))
    utils._assert_same_dim(*params)

    diff = (u - v).abs_()
    dist = torch.sum(diff, dim)

    return dist.squeeze()


@utils.tensorfy(0, 1)
def braycurtis_distance(u, v, dim=1, eps=1e-8):
    """The Bray-Curtis is defined as:

        :math:`braycurtis\_distance(u, v) = \dfrac{\sum_i|u_i - v_i|}{max(\sum_i|u_i + v_i|, eps)}`

    Args:
        u (torch.Tensor): tensor 1
        v (torch.Tensor): tensor 2
        dim (int): the dimension to compare the vectors.
        eps (:obj:`float`): small value to avoid division by zero.

    Shape:
        u (\*, D, \*): where D is the provided dimension ``dim``.
        v (\*, D, \*): where D is the provided dimension ``dim``. ``u`` and ``v`` must have the same shape.
        output (\*, \*): same as input, withoud the ``dim`` dimension, and squeezed.

    Example::

        >>> x1 = torch.Tensor([1, 2, 3])
        >>> x2 = torch.Tensor([2, 2, 5])
        >>> braycurtis_distance(x1, x2, dim=0)
         0.2000
        [torch.FloatTensor of size 1]

    References:
        - http://people.revoledu.com/kardi/tutorial/Similarity/BrayCurtisDistance.html
    """
    params = ((u, 'u'), (v, 'v'))
    utils._assert_same_dim(*params)

    diff = (u - v).abs_()
    wide = (u + v).abs_()

    dist1 = torch.sum(diff, dim)
    dist2 = torch.sum(wide, dim).clamp_(min=eps)

    dist = dist1 / dist2

    dist.squeeze_()

    return dist


@utils.tensorfy(0, 1)
def canberra_distance(u, v, dim=1, eps=1e-8):
    """The Canberra distance between two vectors is defined as:

        :math:`canberra\_distance(u, v) = \sum_i\dfrac{|u_i - v_i|}{|u_i| + |v_i|}`

    Args:
        u (torch.Tensor): tensor 1
        v (torch.Tensor): tensor 2
        dim (int): the dimension to compare the vectors.
        eps (:obj:`float`): small value to avoid division by zero.

    Shape:
        u (\*, D, \*): where D is the provided dimension ``dim``.
        v (\*, D, \*): where D is the provided dimension ``dim``. ``u`` and ``v`` must have the same shape.
        output (\*, \*): same as input, withoud the ``dim`` dimension, and squeezed.

    Example::

        >>> x1 = torch.Tensor([1, 2, 3])
        >>> x2 = torch.Tensor([2, 2, 5])
        >>> canberra_distance(x1, x2, dim=0)
         0.5833
        [torch.FloatTensor of size 1]

    References:
        - https://en.wikipedia.org/wiki/Canberra_distance
    """
    params = ((u, 'u'), (v, 'v'))
    utils._assert_same_dim(*params)

    diff = (u - v).abs_()
    u.abs_()
    v.abs_()
    s = u + v + eps

    coef = diff / s

    dist = torch.sum(coef, dim)

    return dist.squeeze()


@utils.tensorfy(0, 1)
def chebyshev_distance(u, v, dim=1):
    """The Chebyshev distance (or Tchebychev distance), maximum metric,
    or :math:`\ell\_\inf` (the infinity) norm between two vectors is defined as:

        :math:`chebyshev\_distance(u, v) = max_i|u_i - v_i|`

    Args:
        u (torch.Tensor): tensor 1
        v (torch.Tensor): tensor 2
        dim (int): the dimension to compare the vectors.

    Shape:
        u (\*, D, \*): where D is the provided dimension ``dim``.
        v (\*, D, \*): where D is the provided dimension ``dim``. ``u`` and ``v`` must have the same shape.
        output (\*, \*): same as input, withoud the ``dim`` dimension, and squeezed.

    Example::

        >>> x1 = torch.Tensor([1, 2, 3])
        >>> x2 = torch.Tensor([2, 2, 5])
        >>> chebyshev_distance(x1, x2, dim=0)
         2
        [torch.FloatTensor of size 1]

    References:
        - https://en.wikipedia.org/wiki/Lp_space
        - https://en.wikipedia.org/wiki/Chebyshev_distance
    """
    params = ((u, 'u'), (v, 'v'))
    utils._assert_same_dim(*params)

    diff = (u - v).abs_()

    dist, _ = torch.max(diff, dim)

    return dist.squeeze()
