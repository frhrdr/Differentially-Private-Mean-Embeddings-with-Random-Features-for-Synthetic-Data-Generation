import torch as pt
# implementation of non-appriximate mmd loss mostly for debugging purposes


def get_squared_dist(x, y=None):
    """ This function calculates the pairwise distance between x and x, x and y, y and y
    Warning: when x, y has mean far away from zero, the distance calculation is not accurate; use get_dist_ref instead
    :param x: batch_size-by-d matrix
    :param y: batch_size-by-d matrix
    :return:
    """

    xxt = pt.mm(x, x.t())  # (bs, bs)
    xyt = pt.mm(x, y.t())  # (bs, bs)
    yyt = pt.mm(y, y.t())  # (bs, bs)

    dx = pt.diag(xxt)    # (bs)
    dy = pt.diag(yyt)

    dist_xx = pt.nn.functional.relu(dx[:, None] - 2.0 * xxt + dx[None, :])
    dist_xy = pt.nn.functional.relu(dx[:, None] - 2.0 * xyt + dy[None, :])
    dist_yy = pt.nn.functional.relu(dy[:, None] - 2.0 * yyt + dy[None, :])

    return dist_xx, dist_xy, dist_yy


def mmd_g(dist_xx, dist_xy, dist_yy, batch_size, sigma=1.0, upper_bound=None, lower_bound=None):
  """This function calculates the maximum mean discrepancy with Gaussian distribution kernel
  The kernel is taken from following paper:
  Li, C.-L., Chang, W.-C., Cheng, Y., Yang, Y., & Póczos, B. (2017).
  MMD GAN: Towards Deeper Understanding of Moment Matching Network.
  :param dist_xx:
  :param dist_xy:
  :param dist_yy:
  :param batch_size:
  :param sigma:
  :param upper_bound: bounds for pairwise distance in mmd-g.
  :param lower_bound:
  :return:
  """

  if lower_bound is None:
    k_xx = pt.exp(-dist_xx / (2.0 * sigma ** 2))
    k_yy = pt.exp(-dist_yy / (2.0 * sigma ** 2))
  else:
    k_xx = pt.exp(-pt.max(dist_xx, lower_bound) / (2.0 * sigma ** 2))
    k_yy = pt.exp(-pt.max(dist_yy, lower_bound) / (2.0 * sigma ** 2))

  if upper_bound is None:
    k_xy = pt.exp(-dist_xy / (2.0 * sigma ** 2))
  else:
    k_xy = pt.exp(-pt.min(dist_xy, upper_bound) / (2.0 * sigma ** 2))

  # m = tf.constant(batch_size, tf.float32)
  e_kxx = matrix_mean_wo_diagonal(k_xx, batch_size)
  e_kxy = matrix_mean_wo_diagonal(k_xy, batch_size)
  e_kyy = matrix_mean_wo_diagonal(k_yy, batch_size)

  mmd = e_kxx + e_kyy - 2.0 * e_kxy
  return mmd


def matrix_mean_wo_diagonal(matrix, num_row, num_col=None):
    """ This function calculates the mean of the matrix elements not in the diagonal
    2018.4.9 - replace tf.diag_part with tf.matrix_diag_part
    tf.matrix_diag_part can be used for rectangle matrix while tf.diag_part can only be used for square matrix
    :param matrix:
    :param num_row:
    :type num_row: float
    :param num_col:
    :type num_col: float
    :param name:
    :return:
    """
    diff = pt.sum(matrix) - pt.sum((matrix.diag()))
    normalizer = num_row * (num_row - 1.0) if num_col is None else (num_row * num_col - min(num_col, num_row))
    return diff / normalizer