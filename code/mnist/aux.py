import os
import numpy as np
import torch as pt
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def rff_gauss(x, w):
  """
  this is a Pytorch version of anon's code for RFFKGauss
  Fourier transform formula from http://mathworld.wolfram.com/FourierTransformGaussian.html
  """

  xwt = pt.mm(x, w.t())
  z_1 = pt.cos(xwt)
  z_2 = pt.sin(xwt)

  z = pt.cat((z_1, z_2), 1) / pt.sqrt(pt.tensor(w.shape[1]).to(pt.float32))  # w.shape[1] == n_features / 2
  return z


def expand_vector(v, tgt_vec):
  tgt_dims = len(tgt_vec.shape)
  if tgt_dims == 2:
    return v[:, None]
  elif tgt_dims == 3:
    return v[:, None, None]
  elif tgt_dims == 4:
    return v[:, None, None, None]
  elif tgt_dims == 5:
    return v[:, None, None, None, None]
  elif tgt_dims == 6:
    return v[:, None, None, None, None, None]
  else:
    return ValueError


def get_mnist_dataloaders(batch_size, test_batch_size, use_cuda, normalize=True,
                          dataset='digits', data_dir='data'):
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  transforms_list = [transforms.ToTensor()]
  if dataset == 'digits':
    if normalize:
      mnist_mean = 0.1307
      mnist_sdev = 0.3081
      transforms_list.append(transforms.Normalize((mnist_mean,), (mnist_sdev,)))
    prep_transforms = transforms.Compose(transforms_list)
    trn_data = datasets.MNIST(data_dir, train=True, download=True, transform=prep_transforms)
    tst_data = datasets.MNIST(data_dir, train=False, transform=prep_transforms)
    train_loader = pt.utils.data.DataLoader(trn_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = pt.utils.data.DataLoader(tst_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  elif dataset == 'fashion':
    assert not normalize
    prep_transforms = transforms.Compose(transforms_list)
    trn_data = datasets.FashionMNIST(data_dir, train=True, download=True, transform=prep_transforms)
    tst_data = datasets.FashionMNIST(data_dir, train=False, transform=prep_transforms)
    train_loader = pt.utils.data.DataLoader(trn_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = pt.utils.data.DataLoader(tst_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  return train_loader, test_loader


def plot_mnist_batch(mnist_mat, n_rows, n_cols, save_path, denorm=True, save_raw=True):
  bs = mnist_mat.shape[0]
  n_to_fill = n_rows * n_cols - bs
  mnist_mat = np.reshape(mnist_mat, (bs, 28, 28))
  fill_mat = np.zeros((n_to_fill, 28, 28))
  mnist_mat = np.concatenate([mnist_mat, fill_mat])
  mnist_mat_as_list = [np.split(mnist_mat[n_rows*i:n_rows*(i+1)], n_rows) for i in range(n_cols)]
  mnist_mat_flat = np.concatenate([np.concatenate(k, axis=1).squeeze() for k in mnist_mat_as_list], axis=1)

  if denorm:
     mnist_mat_flat = denormalize(mnist_mat_flat)
  save_img(save_path + '.png', mnist_mat_flat)
  if save_raw:
    np.save(save_path + '_raw.npy', mnist_mat_flat)


def save_gen_labels(label_mat, n_rows, n_cols, save_path, save_raw=True):
  if save_raw:
    np.save(save_path + '_raw.npy', label_mat)
  max_labels = np.argmax(label_mat, axis=1)
  bs = label_mat.shape[0]
  n_to_fill = n_rows * n_cols - bs
  fill_mat = np.zeros((n_to_fill,), dtype=np.int) - 1
  max_labels = np.concatenate([max_labels, fill_mat])
  max_labels_flat = np.stack(np.split(max_labels, n_cols, axis=0), axis=1)
  with open(save_path + '_max_labels.txt', mode='w+') as file:
    file.write(str(max_labels_flat))


def denormalize(mnist_mat):
  mnist_mean = 0.1307
  mnist_sdev = 0.3081
  return np.clip(mnist_mat * mnist_sdev + mnist_mean, a_min=0., a_max=1.)


def save_img(save_file, img):
  plt.imsave(save_file, img, cmap=cm.gray, vmin=0., vmax=1.)


def meddistance(x, subsample=None, mean_on_fail=True):
  """
  Compute the median of pairwise distances (not distance squared) of points
  in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

  Parameters
  ----------
  x : n x d numpy array
  mean_on_fail: True/False. If True, use the mean when the median distance is 0.
      This can happen especially, when the data are discrete e.g., 0/1, and
      there are more slightly more 0 than 1. In this case, the m

  Return
  ------
  median distance
  """
  if subsample is None:
    d = dist_matrix(x, x)
    itri = np.tril_indices(d.shape[0], -1)
    tri = d[itri]
    med = np.median(tri)
    if med <= 0:
      # use the mean
      return np.mean(tri)
    return med

  else:
    assert subsample > 0
    rand_state = np.random.get_state()
    np.random.seed(9827)
    n = x.shape[0]
    ind = np.random.choice(n, min(subsample, n), replace=False)
    np.random.set_state(rand_state)
    # recursion just one
    return meddistance(x[ind, :], None, mean_on_fail)


def dist_matrix(x, y):
  """
  Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
  """
  sx = np.sum(x ** 2, 1)
  sy = np.sum(y ** 2, 1)
  d2 = sx[:, np.newaxis] - 2.0 * x.dot(y.T) + sy[np.newaxis, :]
  # to prevent numerical errors from taking sqrt of negative numbers
  d2[d2 < 0] = 0
  d = np.sqrt(d2)
  return d


def log_args(log_dir, args):
  """ print and save all args """
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  with open(os.path.join(log_dir, 'args_log'), 'w') as f:
    lines = [' • {:<25}- {}\n'.format(key, val) for key, val in vars(args).items()]
    f.writelines(lines)
    for line in lines:
      print(line.rstrip())
  print('-------------------------------------------')


def parse_n_hid(n_hid, conv=False):
  """
  make sure conversion is the same everywhere
  """
  if conv:  #
    return [(s[0], [int(k) for k in s[1:].split('-')]) for s in n_hid.split(',')]
  else:  # fully connected case: just a list of linear ops
    return tuple([int(k) for k in n_hid.split(',')])


def flat_data(data, labels, device, n_labels=10, add_label=False):
  # then make one_hot
  bs = data.shape[0]
  if add_label:
    gen_one_hots = pt.zeros(bs, n_labels, device=device)
    gen_one_hots.scatter_(1, labels[:, None], 1)
    labels = gen_one_hots
    return pt.cat([pt.reshape(data, (bs, -1)), labels], dim=1)
  else:
    return pt.reshape(data, (bs, -1))